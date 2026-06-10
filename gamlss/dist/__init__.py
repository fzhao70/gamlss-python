"""gamlss.dist: distribution families and their d/p/q/r functions.

Mirrors the R package gamlss.dist. Each family is a callable returning
a GamlssFamily (e.g. ``NO()``), accompanied by the four functions
``dNO``, ``pNO``, ``qNO``, ``rNO``.
"""

from .NO import NO, NO2, dNO, pNO, qNO, rNO, dNO2, pNO2, qNO2, rNO2
from .GA import GA, dGA, pGA, qGA, rGA
from .PO import PO, dPO, pPO, qPO, rPO
from .EXP import EXP, dEXP, pEXP, qEXP, rEXP
from .IG import IG, dIG, pIG, qIG, rIG
from .GU import GU, RG, dGU, pGU, qGU, rGU, dRG, pRG, qRG, rRG
from .LO import LO, LOGNO, dLO, pLO, qLO, rLO, dLOGNO, pLOGNO, qLOGNO, rLOGNO
from .TF import TF, dTF, pTF, qTF, rTF
from .WEI import WEI, WEI3, dWEI, pWEI, qWEI, rWEI, dWEI3, pWEI3, qWEI3, rWEI3
from .PE import PE, dPE, pPE, qPE, rPE
from .BCCG import (BCCG, BCCGo, dBCCG, pBCCG, qBCCG, rBCCG,
                   dBCCGo, pBCCGo, qBCCGo, rBCCGo)
from .BCT import BCT, BCTo, dBCT, pBCT, qBCT, rBCT, dBCTo, pBCTo, qBCTo, rBCTo
from .BCPE import (BCPE, BCPEo, dBCPE, pBCPE, qBCPE, rBCPE,
                   dBCPEo, pBCPEo, qBCPEo, rBCPEo)
from .BI import BI, dBI, pBI, qBI, rBI
from .BB import BB, dBB, pBB, qBB, rBB
from .ZABI import ZABI, dZABI, pZABI, qZABI, rZABI
from .ZIBI import ZIBI, dZIBI, pZIBI, qZIBI, rZIBI
from .BE import BE, dBE, pBE, qBE, rBE
from .BEo import BEo, dBEo, pBEo, qBEo, rBEo
from .LG import LG, dLG, pLG, qLG, rLG
from .JSU import JSU, dJSU, pJSU, qJSU, rJSU
from .JSUo import JSUo, dJSUo, pJSUo, qJSUo, rJSUo
from .GG import GG, dGG, pGG, qGG, rGG
from .SHASHo import SHASHo, dSHASHo, pSHASHo, qSHASHo, rSHASHo
from .SN1 import SN1, dSN1, pSN1, qSN1, rSN1
from .IGA import IGAMMA, dIGAMMA, pIGAMMA, qIGAMMA, rIGAMMA
from .NBI import NBI, dNBI, pNBI, qNBI, rNBI
from .NBII import NBII, dNBII, pNBII, qNBII, rNBII
from .GEOM import GEOM, dGEOM, pGEOM, qGEOM, rGEOM
from .PIG import PIG, dPIG, pPIG, qPIG, rPIG
from .ZIP import ZIP, dZIP, pZIP, qZIP, rZIP
from .ZIP2 import ZIP2, dZIP2, pZIP2, qZIP2, rZIP2
from .ZINBI import ZINBI, dZINBI, pZINBI, qZINBI, rZINBI
from .ZANBI import ZANBI, dZANBI, pZANBI, qZANBI, rZANBI

__all__ = [n for n in dir() if not n.startswith("_")]
