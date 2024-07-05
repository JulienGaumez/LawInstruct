from datasets import load_dataset

import instruction_manager
from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType


class LLeQA(AbstractDataset):

    def __init__(self):
        super().__init__(
            "LLeQA",
            "https://huggingface.co/datasets/maastrichtlawtech/lleqa")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        task_type = TaskType.QUESTION_ANSWERING
        jurisdiction = Jurisdiction.BELGIUM
        answer_language = "fr"
        prompt_language = "fr"

        df = load_dataset('rcds/swiss_court_view_generation', 'main', split='train') #Needs to be edited, waiting on access...
        for example in df:
            subset = "longform_legal_qa"
            instruction, instruction_language = instructions.sample(subset)
            prompt = f"Q: {example['question']}"
            answer = f"A: {example['answer']}"
            yield self.build_data_point(instruction_language, prompt_language, answer_language,
                                        instruction, prompt, answer, task_type, jurisdiction, subset)
