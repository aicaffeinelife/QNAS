from qiskit.circuit.library import *


ROT_OPS = ['rx', 'ry', 'rz', 'h', 'x', 'y',
           'z']

ENT_OPS = ['cx', 'cy', 'cz', 'crx', 'cry', 'crz', 'rzz', 
           'rxx', 'ryy']

# SINGLE_OPS = {
#     'X': XGate, 'Y': YGate, 'Z': ZGate,
#     'RX': RXGate, 'RY': RYGate, 'RZ': RZGate,
#     'H': HGate
# }

# # 'SWAP': SwapGate,
# # ENTANGLEMENT_OPS = {
   
# # }

# MULTI_OPS = {
#     'XX': RXXGate, 'YY': RYYGate, 'ZZ': RZZGate,
#     'CX': CXGate,  'CY': CYGate, 'CZ': CZGate, 
#     'CRX': CRXGate, 'CRY': CRYGate, 'CRZ': CRZGate,
# }

# PARAM_OPS = ['RX', 'RY', 'RZ', 'CRX', 'CRY', 'CRZ', 'XX', 'YY', 'ZZ']
