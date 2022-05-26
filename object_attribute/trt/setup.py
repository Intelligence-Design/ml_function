import os
import setuptools


def load_requires_from_file(fname):
    if not os.path.exists(fname):
        print(f'Pass: {fname}')
        return []
    return [pkg.strip() for pkg in open(fname, 'r')]


def load_links_from_file(filepath):
    res = []
    with open(filepath) as fp:
        for pkg_name in fp.readlines():
            if "git+ssh" in pkg_name:
                res.append(pkg_name[pkg_name.find("git+ssh"):].strip())
    return res


if __name__ == '__main__':
    setuptools.setup(
        name="object-attribute-trt",
        version="0.0.1",
        url="https://github.com/Intelligence-Design/ml_function/object_attribute/trt",
        install_requires=load_requires_from_file('requirements.txt'),
        dependency_links=load_links_from_file('requirements.txt'),
        author="id",
        author_email="",
        description="ml_function/object_attribute/trt",
        long_description="ml_function/object_attribute/trt",
        long_description_content_type="text/markdown",
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3.6.9",
            "No License :: No Approved :: No License",
            "Operating System :: OS Independent",
        ],
        include_package_data=True,
    )
