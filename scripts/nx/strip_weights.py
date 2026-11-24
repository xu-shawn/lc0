#!/usr/bin/env python3
"""Write a copy of a network file without its weights (lc0 only reads the input format from it; cuda-nx loads the
weights from the blob written by nx_export.py).

    strip_weights.py NET.pb.gz OUT.pb.gz
"""
import gzip, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import net_pb2

net = net_pb2.Net()
net.ParseFromString(gzip.open(sys.argv[1]).read())
net.ClearField("weights")
with gzip.open(sys.argv[2], "wb") as f:
    f.write(net.SerializeToString())
