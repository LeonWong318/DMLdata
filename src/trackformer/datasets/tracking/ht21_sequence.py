"""
HT21 sequence dataset.
"""

from .mot17_sequence import MOT17Sequence


class HT21Sequence(MOT17Sequence):
    """Multiple Object Tracking (HT21) Dataset.

    This dataloader is designed so that it can handle only one sequence,
    if more have to be handled one should inherit from this class.
    """
    data_folder = 'HT21'
