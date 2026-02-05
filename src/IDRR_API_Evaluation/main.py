from utils_zp import *
from IDRR_data import *


class IDRRAPIEvaluation:
    def __init__(self, idrrdfs:IDRRDataFrames):
        self.idrrdfs = idrrdfs
        pass

    @classmethod
    def input_prompt(cls, sample:IDRRDataSample):
        return f'''
Argument1: {sample.arg1}

Argument2: {sample.arg2}

What's the relation between the given two text segments?
A. Comparison
B. Contingency
C. Expansion
D. Temporal

Output only the letter corresponding to your choice: `A`, `B`, `C`, or `D`.

'''.strip()

    # @classmethod
    def output_to_labelid(self, output:str):
        output = output.lower()
        if output in 'abcd':
            return 'abcd'.index(output)
        return -1
        output = self.idrrdfs.label_to_id('Comparison')
        print(output)

    @classmethod
    def cal_metrics(cls, preds: List[int], labels: List[int]) -> Dict[str, float]:
        """
        Calculate accuracy and macro F1-score
        
        Args:
            preds: List of predicted labels
            labels: List of ground truth labels
            
        Returns:
            Dictionary containing 'accuracy' and 'macro_f1' scores
        """
        # Input validation
        if len(preds) != len(labels):
            raise ValueError(f"Length mismatch: preds({len(preds)}) != labels({len(labels)})")
        
        if len(preds) == 0:
            return {"accuracy": 0.0, "macro_f1": 0.0}
        
        # Calculate accuracy
        correct = sum(p == l for p, l in zip(preds, labels))
        accuracy = correct / len(preds)
        
        # Get all unique classes
        all_classes = set(labels)
        
        # Initialize metrics
        f1_scores = []
        
        for class_label in all_classes:
            # Calculate TP, FP, FN for this class
            tp = sum((p == class_label) and (l == class_label) for p, l in zip(preds, labels))
            fp = sum((p == class_label) and (l != class_label) for p, l in zip(preds, labels))
            fn = sum((p != class_label) and (l == class_label) for p, l in zip(preds, labels))
            
            # Calculate precision and recall
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # Calculate F1-score
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
            
            f1_scores.append(f1)
        
        # Calculate macro F1 (average of F1 scores for all classes)
        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
        
        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1
        }

    def evaluate(
        self, 
        model:Callable, 
        result_dir:Union[str,path], 
        input_prompt_func:Callable=None,
    ):
        if input_prompt_func is None:
            input_prompt_func = self.input_prompt
        preds, labels = [], []
        unexpected_output = []
        for sample in tqdm.tqdm(self.idrrdfs.test_di):
            _input = input_prompt_func(sample)
            _output = model(_input)
            _pred = self.output_to_labelid(output=_output)
            if _pred == -1:
                unexpected_output.append(_output)
            preds.append(_pred)
            labels.append(sample.label11id)
        metrics = self.cal_metrics(preds=preds, labels=labels)
        print(metrics)
        auto_dump(metrics, path(result_dir, 'result.json'))
        if unexpected_output:
            auto_dump(unexpected_output, path(result_dir, 'unexpected_output.json'))
        return metrics


# if __name__ == '__main__':
#     idrrdfs = IDRRDataFrames(
#         data_name='pdtb3',
#         data_level='top',
#         data_relation='Implicit',
#         data_path='/root/autodl-fs/IDRR_data/data/pdtb3.p2.csv',
#     )
#     APIEvaluation(idrrdfs=idrrdfs).output_to_labelid(1)