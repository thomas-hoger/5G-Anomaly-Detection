from networkanomalydetection.core.trace_dissection.dissect_packet import dissect_packets


def trace_dissection(pkt_files: dict, banned_features: list[str], label_dataframe_files: dict):

    dissected_files = {}

    for file, pkt_loader in pkt_files.items():

        csv_file   = file.replace("pcap","csv")
        csv_loader = label_dataframe_files[csv_file]

        json_file = file.replace("pcap","json")

        dissected_files[json_file] =  dissect_packets(pkt_loader(), banned_features, csv_loader())

    return dissected_files
