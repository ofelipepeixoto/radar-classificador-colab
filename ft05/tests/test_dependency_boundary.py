"""Do not expose the known folder-builder advisory through this text exercise."""
import ast
import unittest
from pathlib import Path
from ft05 import validate


class DependencyBoundaryTests(unittest.TestCase):
    def test_folder_metadata_is_rejected_as_training_input(self):
        row = {'id': 'fiction', 'group_id': 'fiction', 'synthetic': True, 'reviewed': False,
               'messages': [{'role': 'user', 'content': 'Mensagem fictícia'},
                            {'role': 'assistant', 'content': '{"categoria":"outro","urgencia":"baixa"}'}],
               'file_name': '../../private.txt'}
        with self.assertRaisesRegex(ValueError, 'invalid_record'): validate([row])

    def test_dataset_boundary_allows_only_in_memory_constructor(self):
        # Prevent later changes from adding the affected data loaders unnoticed.
        root = Path(__file__).resolve().parents[1]
        for file in root.glob('*.py'):
            tree = ast.parse(file.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == 'datasets':
                    self.assertEqual([name.name for name in node.names], ['Dataset'], str(file))
                if isinstance(node, ast.Call):
                    name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', '')
                    self.assertNotIn(name, {'load_dataset', 'save_to_disk', 'push_to_hub'}, str(file))


if __name__ == '__main__': unittest.main()
