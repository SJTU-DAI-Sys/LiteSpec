"""DFlash2 training registration."""

from ._dflash_common import DFlashFamilyTrainingAdapter

ADAPTER = DFlashFamilyTrainingAdapter("dflash2", "DFlash2DraftModel")


def parse_args(*, mode=None, argv=None):
    from ._dflash_common import parse_dflash_family_args

    return parse_dflash_family_args("dflash2", argv)
