import abc
from rdkit import Chem
from rdkit.Chem import Descriptors


class BaseObjective(abc.ABC):
    """Abstract base class for prodrugs; must have the calculate method implemented."""

    @abc.abstractmethod
    def calculate(self, generated_smiles: str, parent_smiles: str) -> dict:
        """Calculate the objective score for a generated prodrug molecule.

        Args:
            generated_smiles (str): SMILES string of the new generated prodrug molecule.
            parent_smiles (str): SMILES string of the parent drug molecule.
        Returns:
            dict: A dictionary containing the total score and individual component scores.
        """
        pass