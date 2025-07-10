from agents.crl import CRLAgent
from agents.gcbc import GCBCAgent
from agents.gciql import GCIQLAgent
from agents.gcivl import GCIVLAgent
from agents.hiql import HIQLAgent
from agents.hiql_ddpgbc import HIQLDDPGBCAgent
from agents.hiql_ddpgbc_orig import HIQLDDPGBCOGAgent
from agents.qrl import QRLAgent
from agents.sac import SACAgent

agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    hiql_ddpgbc=HIQLDDPGBCAgent,
    hiql_ddpgbc_og=HIQLDDPGBCOGAgent,
    qrl=QRLAgent,
    sac=SACAgent,
)
