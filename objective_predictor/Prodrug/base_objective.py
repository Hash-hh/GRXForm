import abc
from rdkit import Chem


class BaseObjective(abc.ABC):
    """Abstract base class for prodrugs; must have the calculate method implemented."""

    @abc.abstractmethod
    def calculate(self, generated_smiles: Chem.Mol, parent_smiles: Chem.Mol) -> dict:
        """Calculate the objective score for a generated prodrug molecule.

        Args:
            generated_smiles (Chem.Mol): SMILES string of the new generated prodrug molecule.
            parent_smiles (Chem.Mol): SMILES string of the parent drug molecule.
        Returns:
            dict: A dictionary containing the total score and individual component scores.
        """
        pass