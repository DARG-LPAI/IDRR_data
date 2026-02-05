from utils_zp import *
from IDRR_data import *
from llm_zp import *
from IDRR_API_Evaluation import IDRRAPIEvaluation


pdtb3top_idrrapieval = IDRRAPIEvaluation(
    IDRRDataFrames(
        'pdtb3', 'top', 'Implicit', '/root/autodl-fs/IDRR_data/data/pdtb3.p2.csv'
    )
)


def qwen3max():
    model_name = 'qwen3_max'
    api = APICalling_zp(
        api_key=api_key_dic['aliyun'],
        model='qwen3-max',
        print_io=False,
    )
    def model_func(query):
        return api.chat_dashscope(query,)
    def input_prompt(sample:IDRRDataSample):
            return f'''
    Argument1: {sample.arg1}

    Argument2: {sample.arg2}

    What's the relation between the given two text segments?
    A. Comparison
    B. Contingency
    C. Expansion
    D. Temporal

    Output only the letter corresponding to your choice: `A`, `B`, `C`, or `D`.
    /no_think

    '''.strip()

    pdtb3top_idrrapieval.evaluate(
        model_func, result_dir=path('/root/autodl-fs/IDRR_data/data', 'api_eval', model_name),
        input_prompt_func=input_prompt
    )


def deepseekchat():
    model_name = 'deepseekchat'
    api = APICalling_zp(
        api_key=api_key_dic['ds'],
        base_url=APIURLName.deepseek,
        model='deepseek-chat',
        # print_io=True,
    )
    def model_func(query):
        return api.chat_openai(query,)
    def input_prompt(sample:IDRRDataSample):
            return f'''
    Argument1: {sample.arg1}

    Argument2: {sample.arg2}

    What's the relation between the given two text segments?
    A. Comparison
    B. Contingency
    C. Expansion
    D. Temporal

    Output only the letter corresponding to your choice: `A`, `B`, `C`, or `D`.
    /no_think

    '''.strip()

    pdtb3top_idrrapieval.evaluate(
        model_func, result_dir=path('/root/autodl-fs/IDRR_data/data', 'api_eval', model_name),
        # input_prompt_func=input_prompt
    )


if __name__ == '__main__':
    # qwen3max()
    deepseekchat()


