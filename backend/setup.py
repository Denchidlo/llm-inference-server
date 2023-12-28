from setuptools import setup 
import functools


extras = {
    'vllm': ['vllm==0.2.3'],
    'tensorrt-llm' : ['transformers==4.33.1', 'tensorrt_llm==0.6.1'],
    'deepspeed': ['deepspeed>=0.12.5'],
    'onnx': ['optimum[onnxruntime-gpu]'],
    'perf_test': ['boto3', 'langchain', 'openpyxl']
}
extras['all'] = functools.reduce(lambda a, b: a + b, [v for v in extras.values()], [])

setup( 
    name='backend', 
    version='0.1', 
    description='Python package for fast LM inference. By default package '
    'supports inference only via transformers framework, '
    f'use extras {list(e for e in extras.keys())} to '
    'import dependencies for framework needed',
    author='', 
    author_email='', 
    packages=['batched_inference'], 
    install_requires=[ 
        'transformers'
    ], 
    extras_require=extras
) 
