import json
import re
import os
import logging
import pyshark
import jwt
from typing import Dict, List, Optional, Any, Tuple, Union
from urllib.parse import parse_qs

logger = logging.getLogger(__name__)

# Code original - exactement comme dans votre fichier
http_type = {
    0: "DATA",
    1: "HEADERS",
    2: "PRIORITY",
    3: "RST_STREAM",
    4: "SETTINGS",
    5: "PUSH_PROMISE",
    6: "PING",
    7: "GOAWAY",
    8: "WINDOW_UPDATE",
    9: "CONTINUATION"
}

regex_imsi = r'imsi-\d{15}'                       # Capture IMSI sous forme "imsi-XXXXXXXXXXXXXXX"
regex_suci = r'suci-\d+-\d+-\d+-\d+-\d+-\d+-\d+'  # Capture SUCI sous forme "suci-X-X-X-X-X-X-X"

class PacketDissector:
    def __init__(self, ban_list: List[str] = None):
        self.ban_list = ban_list or []
        self.packet_count = 0
        self.opened_flux = {}

    def is_request(self, packet, packet_id: int):
        """
        Determines if a given packet is a request or a response and manages the flow state.
        """
        if "streamid" not in packet.http2.field_names:
            return True, None
        
        ip_src, ip_dst = str(packet.ip.src), str(packet.ip.dst)
        streamid = packet.http2.streamid
        request = False
        original_request_id = None
        
        # The message is not a response (i.e the flow have been opened in the other direction): 
        if (ip_dst, ip_src) not in self.opened_flux or streamid not in self.opened_flux[(ip_dst, ip_src)]:
            request = True

            # First message between the 2 entities (in the same flow or not)
            if (ip_src, ip_dst) not in self.opened_flux:
                self.opened_flux[(ip_src, ip_dst)] = {}

            # First request of the same flow 
            if streamid not in self.opened_flux[(ip_src, ip_dst)]:
                self.opened_flux[(ip_src, ip_dst)][streamid] = packet_id
            # POST request with DATA can sometimes be split in 2 requests so we merge them back
            else:
                original_request_id = self.opened_flux[(ip_src, ip_dst)][streamid]

        # Is a response
        else:
            original_request_id = self.opened_flux[(ip_dst, ip_src)][streamid]

        # If it's the end of the stream we remove from the list
        if "flags_end_stream" in packet.http2.field_names:
            if (ip_dst, ip_src) in self.opened_flux:
                if streamid in self.opened_flux[(ip_dst, ip_src)] and packet.http2.flags_end_stream.int_value:
                    del self.opened_flux[(ip_dst, ip_src)][streamid]
                if len(self.opened_flux[(ip_dst, ip_src)]) < 1:
                    del self.opened_flux[(ip_dst, ip_src)]

        return request, original_request_id

    def remove_banned_values(self, d: Union[dict, list], ban_list: list) -> None:
        """
        Recursively traverses a nested dictionary (and lists within it) to remove all occurrences matching a banned item.
        """
        d_copy = d.copy()

        if isinstance(d, dict):
            for key, value in d_copy.items():
                if isinstance(key, str) and key.lower() in ban_list:
                    del d[key]
                else:
                    if isinstance(value, dict) or isinstance(value, list):
                        self.remove_banned_values(value, ban_list)
                    elif isinstance(value, str):
                        if value.lower() in ban_list:
                            del d[key]

        if isinstance(d, list):
            for value in d_copy:
                if isinstance(value, dict) or isinstance(value, list):
                    self.remove_banned_values(value, ban_list)
                elif isinstance(value, str) and value.lower() in ban_list:
                    d.remove(value)

    def replace_imsi(self, d: Union[dict, list]) -> None:
        """
        Recursively traverses a nested dictionary (and lists within it) to replace every value matching an IMSI pattern to the value given by get_imsi
        """
        if isinstance(d, dict):
            for key, value in d.items():
                if isinstance(value, (dict, list)):
                    self.replace_imsi(value)
                elif isinstance(value, str):
                    imsi = self.get_imsi(value)
                    if imsi:
                        d[key] = imsi

        elif isinstance(d, list):
            for i, value in enumerate(d):
                if isinstance(value, (dict, list)):
                    self.replace_imsi(value)
                elif isinstance(value, str):
                    imsi = self.get_imsi(value)
                    if imsi:
                        d[i] = imsi

    def get_imsi(self, text: str) -> str:
        """
        Extracts the IMSI (International Mobile Subscriber Identity) or SUCI (Subscription Concealed Identifier) 
        from the given text using regular expressions.
        """
        id_value = id_type = ""
        if not id_value:
            id_value = re.search(regex_suci, text)
        if not id_value:
            id_value = re.search(regex_imsi, text)
        if id_value:
            id_value = id_value.group().split("-")
            id_type = id_value.pop(0)

            if id_type == "suci":
                supi_type = int(id_value[0])
                if supi_type == 0:   # IMSI
                    id_value = id_value[1] + id_value[2] + id_value[6]
                elif supi_type == 1:  # Network Access Identifier (NAI)
                    id_value = ""
                else:                # Spare values for future use.
                    id_value = ""
            else:
                id_value = id_value[0]

        return id_value

    def decode_jwt(self, token: str):
        """
        Decodes a JSON Web Token (JWT) and returns its header and content.
        """
        try:
            token = token.replace("Bearer ", "")
            header = jwt.get_unverified_header(token)
            content = jwt.decode(token, options={"verify_signature": False})
            return header | content
        except Exception as e:
            logger.warning(f"Erreur décodage JWT: {e}")
            return {}

    def extract_url_encoded(self, data: str):
        try:
            parsed = parse_qs(data)
            for key, value in list(parsed.items()):
                if isinstance(value, list) and len(value) == 1:
                    parsed[key] = value[0]
            return parsed
        except:
            return {}

    def extract_json(self, data: str):
        try:
            loaded = json.loads(data)
            if isinstance(loaded, list):
                if len(loaded) > 1:
                    print("PROBLEME", loaded)
                loaded = loaded[0]
            if not loaded:
                raise ValueError
            return loaded
        except:
            return {}

    def extract_json_mime(self, packet):
        if hasattr(packet.http2, 'json_object'):
            json_data = packet.http2.json_object
            json_dict = json.loads(json_data)
            return json_dict
        return {}

    def update_dictionary(self, dict_1: dict, dict_2: dict) -> dict:
        """Update the first dictionary with the content of the second. Concatenate their values as lists in case of key collision."""
        for key2, value2 in dict_2.items():
            if value2:
                if key2 not in dict_1:
                    dict_1[key2] = value2
                else:
                    for key1, value1 in dict_1.items():
                        if key1 == key2:
                            if (isinstance(value1, list) and value2 not in value1):
                                dict_1[key1] += [dict_2[key2]]
                            elif value1 != value2:
                                dict_1[key1] = [dict_1[key1]] + [dict_2[key2]]
        return dict_1

    def field_unpacking(self, field: str, pkt_json_content: dict) -> dict:
        """If an HTTP packet contain a certain field it must be further jsonified."""
        field_content = pkt_json_content[field]
        if isinstance(field_content, list) and len(field_content) == 1:
            field_content = field_content[0]

        try:
            field_content = json.loads(field_content)

            if field_content:
                if isinstance(field_content, list):
                    for value in field_content:
                        self.update_dictionary(pkt_json_content, value)
                else:
                    self.update_dictionary(pkt_json_content, field_content)

        except ValueError:
            new_field_name = f"{field}_unpacked"
            self.update_dictionary(pkt_json_content, {new_field_name: field_content})

        return pkt_json_content

    def dissect_http2(self, packet) -> list:
        """Dissect HTTP2 Frame packet"""
        dissected_layers = []

        for layer in packet.layers:
            if layer.layer_name == 'http2':
                fields = layer._all_fields
                if "http2.type" in fields:
                    fields["http2.type"] = http_type[int(fields["http2.type"])]
                    content = {}

                    if fields["http2.type"] == "DATA":
                        try:
                            data = layer.get("http2.data.data").binary_value.decode('UTF8', 'replace')
                            self.update_dictionary(content, self.extract_url_encoded(data))
                            self.update_dictionary(content, self.extract_json(data))
                            self.update_dictionary(content, self.extract_json_mime(packet))
                        except:
                            pass

                    elif fields["http2.type"] == "HEADERS":
                        try:
                            headers = {key: val for key, val in fields.items() if ".header" in key}
                            if "http2.headers" in headers:
                                headers.pop("http2.headers")
                            if "http2.header" in headers:
                                headers["header"] = headers.pop("http2.header")
                            headers = {
                                key.replace("http2.header.", "").replace("http2.headers.", "").replace("http2.", ""): val
                                for key, val in headers.items()
                            }
                            self.update_dictionary(content, headers)
                        except:
                            pass

                    if content:
                        # Get the imsi 
                        self.replace_imsi(content)

                        # Decipher the jwt
                        for jwt_key in ["access_token", "authorization"]:
                            if jwt_key in content:
                                jwt_raw = content.pop(jwt_key)
                                if isinstance(jwt_raw, list):
                                    jwt_raw = jwt_raw[0]
                                jwt_decoded = self.decode_jwt(jwt_raw)
                                if jwt_decoded:
                                    content["jwt"] = jwt_decoded

                        # Get the stream for identifying request/response
                        if 'http2.flags.end_stream' in fields:
                            content["stream_response"] = str(fields['http2.flags.end_stream']) == "True"
                            content["stream_id"] = int(fields['http2.streamid'])

                        dissected_layers.append(content)

        return dissected_layers

    def dissect_packet(self, packet) -> list:
        """Dissect application level features in a given packet"""
        all_dissected_layers = []

        # IP 
        if not hasattr(packet, 'ip'):
            return all_dissected_layers

        # Port
        if hasattr(packet, 'tcp'):
            port_src = packet.tcp.srcport
            port_dst = packet.tcp.dstport
        elif hasattr(packet, 'udp'):
            port_src = packet.udp.srcport
            port_dst = packet.udp.dstport
        else:
            return all_dissected_layers

        self.packet_count += 1

        common = {
            "ip_src": str(packet.ip.src),
            "ip_dst": str(packet.ip.dst),
            "port_src": int(port_src),
            "port_dst": int(port_dst),
            "ts": float(packet.sniff_timestamp),
            "packet_id": self.packet_count
        }

        # HTTP2 packets 
        if 'HTTP2' in packet:
            dissected_layers = self.dissect_http2(packet)

            # HTTP2 packets can contain multiple layers
            for dissected_layer in dissected_layers:
                if dissected_layer:
                    if self.ban_list:
                        self.remove_banned_values(dissected_layer, self.ban_list)

                    current_dissected_pkt = {
                        "http2": dissected_layer,
                        "common": common.copy()
                    }

                    is_req, original_request_id = self.is_request(packet, self.packet_count)
                    current_dissected_pkt["common"]["is_request"] = is_req
                    current_dissected_pkt["common"]["original_request_id"] = original_request_id
                    all_dissected_layers.append(current_dissected_pkt)

        return all_dissected_layers

    def update_json_file(self, output_directory: str, node_name: str, node_json_content: dict) -> list:
        """Update the content of ./output/<filename>/extracted/<node_name>.json with new content and return the newly added keys."""
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        node_file = f'{output_directory}/{node_name}.json'

        old_node_json_content = {}
        if os.path.isfile(node_file):
            with open(node_file, 'r') as f:
                old_node_json_content = json.load(f)

        node_json_content = self.update_dictionary(old_node_json_content, node_json_content)
        with open(node_file, 'w') as f:
            json.dump(node_json_content, f, indent=2)


# Fonction principale pour Kedro
def dissect_pcap_file(pcap_file: str, 
                     max_packets: int = 10000,
                     ban_list: List[str] = None) -> List[Dict[str, Any]]:
    """
    Dissèque un fichier PCAP en utilisant le code original adapté
    """
    dissector = PacketDissector(ban_list=ban_list)
    all_packets = []
    
    try:
        capture = pyshark.FileCapture(pcap_file, display_filter='http2')
        
        for i, packet in enumerate(capture):
            if i >= max_packets:
                break
                
            dissected = dissector.dissect_packet(packet)
            all_packets.extend(dissected)
            
            if i % 1000 == 0:
                logger.info(f"Traité {i} paquets de {pcap_file}")
        
        capture.close()
        logger.info(f"Dissection terminée: {len(all_packets)} paquets HTTP2 extraits de {pcap_file}")
        
    except Exception as e:
        logger.error(f"Erreur lors de la dissection de {pcap_file}: {e}")
    
    return all_packets