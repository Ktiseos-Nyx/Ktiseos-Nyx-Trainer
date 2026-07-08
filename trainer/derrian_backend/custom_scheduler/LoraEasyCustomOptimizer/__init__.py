
from typing import Dict, List
from LoraEasyCustomOptimizer.utils import OPTIMIZER

from LoraEasyCustomOptimizer.adabelief import AdaBelief
from LoraEasyCustomOptimizer.adagc import AdaGC
from LoraEasyCustomOptimizer.adammini import AdamMini
from LoraEasyCustomOptimizer.adan import Adan
from LoraEasyCustomOptimizer.ademamix import (AdEMAMix, SimplifiedAdEMAMix, SimplifiedAdEMAMixExM)
from LoraEasyCustomOptimizer.adopt import ADOPT
from LoraEasyCustomOptimizer.came import CAME
from LoraEasyCustomOptimizer.compass import Compass, Compass8BitBNB, CompassPlus, CompassADOPT, CompassADOPTMARS, CompassAO
from LoraEasyCustomOptimizer.farmscrop import FARMSCrop, FARMSCropV2
from LoraEasyCustomOptimizer.fcompass import FCompass, FCompassPlus, FCompassADOPT, FCompassADOPTMARS
from LoraEasyCustomOptimizer.fishmonger import FishMonger, FishMonger8BitBNB
from LoraEasyCustomOptimizer.fmarscrop import FMARSCrop, FMARSCropV2, FMARSCropV2ExMachina, FMARSCropV3, FMARSCropV3ExMachina
from LoraEasyCustomOptimizer.galore import GaLore
from LoraEasyCustomOptimizer.gooddog import GOODDOG
from LoraEasyCustomOptimizer.grokfast import GrokFastAdamW
from LoraEasyCustomOptimizer.laprop import LaProp
from LoraEasyCustomOptimizer.lpfadamw import LPFAdamW
from LoraEasyCustomOptimizer.ranger21 import Ranger21
from LoraEasyCustomOptimizer.spam import StableSPAM
from LoraEasyCustomOptimizer.rmsprop import RMSProp, RMSPropADOPT, RMSPropADOPTMARS
from LoraEasyCustomOptimizer.schedulefree import (
    ScheduleFreeWrapper, ADOPTScheduleFree, ADOPTEMAMixScheduleFree, ADOPTNesterovScheduleFree, 
    FADOPTScheduleFree, ADOPTMARSScheduleFree, FADOPTMARSScheduleFree, ADOPTAOScheduleFree
    )

from LoraEasyCustomOptimizer.clybius_experiments import (MomentusCaution, REMASTER)
from LoraEasyCustomOptimizer.scion import SCION
from LoraEasyCustomOptimizer.sgd import SGDSaI
from LoraEasyCustomOptimizer.shampoo import ScalableShampoo
from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree


class _LazyAO:
    """Lazy-loads an AO optimizer class on first instantiation.

    This keeps torchao out of memory unless AdamW8bitAO/4bitAO/fp8AO
    is actually selected — the torchao import chain (adam.py →
    low_bit_optim → torchao.utils) fires only when the optimizer's
    __call__ is invoked, not at LoraEasyCustomOptimizer import time.
    """
    _resolved: dict | None = None

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def __name__(self) -> str:
        return self._name

    @property
    def __module__(self) -> str:
        return "LoraEasyCustomOptimizer.adam"

    @property
    def __qualname__(self) -> str:
        return self._name

    def __call__(self, *args, **kwargs):
        return self._resolve()(*args, **kwargs)

    def _resolve(self):
        if _LazyAO._resolved is None:
            from LoraEasyCustomOptimizer.adam import AdamW8bitAO, AdamW4bitAO, AdamWfp8AO
            _LazyAO._resolved = {
                "AdamW8bitAO": AdamW8bitAO,
                "AdamW4bitAO": AdamW4bitAO,
                "AdamWfp8AO": AdamWfp8AO,
            }
        return _LazyAO._resolved[self._name]


AdamW8bitAO = _LazyAO("AdamW8bitAO")
AdamW4bitAO = _LazyAO("AdamW4bitAO")
AdamWfp8AO = _LazyAO("AdamWfp8AO")
from LoraEasyCustomOptimizer.scorn import SCORN
from LoraEasyCustomOptimizer.scornmachina import SCORNMachina
from LoraEasyCustomOptimizer.mythical import Mythical
from LoraEasyCustomOptimizer.oagopt import OAGOpt
from LoraEasyCustomOptimizer.ocgopt import OCGOpt
from LoraEasyCustomOptimizer.glyph import Glyph
from LoraEasyCustomOptimizer.racs import RACS
from LoraEasyCustomOptimizer.alice import Alice
from LoraEasyCustomOptimizer.fira import Fira
from LoraEasyCustomOptimizer.vsgd import VSGD
from LoraEasyCustomOptimizer.cstableadamw import CStableAdamW
from LoraEasyCustomOptimizer.dehaze import Dehaze
from LoraEasyCustomOptimizer.talon import TALON
from LoraEasyCustomOptimizer.fftdescent import FFTDescent
from LoraEasyCustomOptimizer.scgopt import SCGOpt
from LoraEasyCustomOptimizer.singstate import SingState
from LoraEasyCustomOptimizer.snoo_asgd import SNOO_ASGD
from adv_optm.optim import AdamW_adv, Adopt_adv, Simplified_AdEMAMix as Simplified_AdEMAMix_adv, Lion_adv
from LoraEasyCustomOptimizer.abmog import ABMOG
from LoraEasyCustomOptimizer.bcos import BCOS
from LoraEasyCustomOptimizer.projective_adam import ProjectiveAdam
from LoraEasyCustomOptimizer.wiwiopt import WiwiOpt
from LoraEasyCustomOptimizer.adam_kahan import AdamW8bitKahan
from LoraEasyCustomOptimizer.cascade import CASCADE
from LoraEasyCustomOptimizer.radam_schedulefree import RAdamScheduleFree
from LoraEasyCustomOptimizer.ocgoptv2 import OCGOptV2

OPTIMIZER_LIST: List[OPTIMIZER] = [
    ABMOG,
    AdamW8bitKahan,
    ADOPT,
    ADOPTAOScheduleFree,
    ADOPTEMAMixScheduleFree,
    ADOPTMARSScheduleFree,
    ADOPTNesterovScheduleFree,
    ADOPTScheduleFree,
    AdEMAMix,
    AdaBelief,
    AdaGC,
    AdamMini,
    Adan,
    AdamW_adv,
    AdamW4bitAO,
    AdamW8bitAO,
    AdamWfp8AO,
    Adopt_adv,
    Alice,
    BCOS,
    CAME,
    CASCADE,
    Compass,
    CompassAO,
    Compass8BitBNB,
    CompassADOPT,
    CompassADOPTMARS,
    CompassPlus,
    CStableAdamW,
    Dehaze,
    FADOPTMARSScheduleFree,
    FADOPTScheduleFree,
    FARMSCrop,
    FARMSCropV2,
    FCompass,
    FCompassADOPT,
    FCompassADOPTMARS,
    FCompassPlus,
    Fira,
    FMARSCrop,
    FMARSCropV2,
    FMARSCropV2ExMachina,
    FMARSCropV3,
    FMARSCropV3ExMachina,
    FishMonger,
    FishMonger8BitBNB,
    FFTDescent,
    GaLore,
    Glyph,
    GOODDOG,
    GrokFastAdamW,
    LPFAdamW,
    LaProp,
    Lion_adv,
    MomentusCaution,
    Mythical,
    OAGOpt,
    OCGOpt,
    OCGOptV2,
    ProdigyPlusScheduleFree,
    ProjectiveAdam,
    RACS,
    REMASTER,
    RMSProp,
    RMSPropADOPT,
    RMSPropADOPTMARS,
    RAdamScheduleFree,
    Ranger21,
    SCION,
    SGDSaI,
    ScalableShampoo,
    SCGOpt,
    ScheduleFreeWrapper,
    SCORN,
    SCORNMachina,
    Simplified_AdEMAMix_adv,
    SimplifiedAdEMAMix,
    SimplifiedAdEMAMixExM,
    SingState,
    SNOO_ASGD,
    StableSPAM,
    TALON,
    VSGD,
    WiwiOpt,
]

OPTIMIZERS: Dict[str, OPTIMIZER] = {str(f"{optimizer.__name__}".lower()): optimizer for optimizer in OPTIMIZER_LIST}