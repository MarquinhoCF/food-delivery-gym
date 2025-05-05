from setuptools import setup, find_packages

setup(
    name='food_delivery_gym',
    version='0.1.0',
    packages=find_packages(where="src"),
    package_dir={'': 'src'},
    include_package_data=True,
    package_data={
        "main": ["scenarios/*.json"],
    },
    install_requires=[
        'simpy',
        'matplotlib',
        'pygame',
        'gymnasium',
        'numpy'
    ],
    author="Marcos Carvalho Ferreira",
    description="Ambiente Gymnasium customizado para simulação de entrega de comida",
    keywords="gymnasium food delivery reinforcement-learning",
    url="https://github.com/MarquinhoCF/food-delivery-simulator",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
