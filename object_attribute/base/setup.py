import os
import setuptools


def load_requires_from_file(fname):
    if not os.path.exists(fname):
        print(f'Pass: {fname}')
        return []
    return [pkg.strip() for pkg in open(fname, 'r')]


if __name__ == '__main__':
    setuptools.setup(
        name="object-attribute-base",
        version="0.0.1",
        url="https://github.com/Intelligence-Design/ml_function/object_attribute/base",
        install_requires=load_requires_from_file('requirements.txt'),
        author="id",
        author_email="",
        description="ml_function/object_attribute/base",
        long_description="ml_function/object_attribute/base",
        long_description_content_type="text/markdown",
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3.8.7",
            "No License :: No Approved :: No License",
            "Operating System :: OS Independent",
        ],
        include_package_data=True,
    )
