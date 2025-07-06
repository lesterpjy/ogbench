from agents.crl import CRLAgent
from agents.gcbc import GCBCAgent
from agents.gciql import GCIQLAgent
from agents.gcivl import GCIVLAgent
from agents.hiql import HIQLAgent
from agents.hiql_ddpgbc import HIQLDDPGBCAgent
from agents.qrl import QRLAgent
from agents.sac import SACAgent

agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    hiql_ddpgbc=HIQLDDPGBCAgent,
    qrl=QRLAgent,
    sac=SACAgent,
)
