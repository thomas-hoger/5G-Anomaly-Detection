import pyshark

cap = pyshark.FileCapture('data/pre_process/trace_clean/5GCTD.pcap')
i = 0
for packet in cap :
    i+=1
    # if "HTTP2" in packet:
    if i == 6:
        print(packet)
        break