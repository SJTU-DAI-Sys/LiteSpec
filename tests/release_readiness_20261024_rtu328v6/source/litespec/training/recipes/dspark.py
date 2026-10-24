"""DSpark training registration."""

from ._dflash_common import DFlashFamilyTrainingAdapter

ADAPTER = DFlashFamilyTrainingAdapter("dspark", "DSparkDraftModel")


def parse_args(*, mode=None, argv=None):
    from ._dflash_common import parse_dflash_family_args

    return parse_dflash_family_args("dspark", argv)
