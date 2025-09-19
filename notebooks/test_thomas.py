import pyshark
import json

i = 0
fields = {}
cap = pyshark.FileCapture('data/pre_process/trace_clean/benign.pcap')
for packet in cap :

    if "nas_eps" in packet or "nas-5gs" in packet or "nas_esm" in packet:
        # print(packet.gtp)
        print(packet)

        # for key,value in packet.gtp._all_fields.items():

        #     if key in fields:
        #         fields[key].append(value)
        #     else :
        #         fields[key] = [value]

    i += 1
    if not i % 1000 : print(i/1000)

# with open('./notebooks/unique_gtp.json', 'w') as f:
#     json.dump(fields, f, indent=2)