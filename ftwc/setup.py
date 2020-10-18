#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='ftwc',
    version='0.0.0',
    description='RL Agents for First TextWorld Challenge',
    author='GStrazds',
    author_email='gstrazds@hotmail.com',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='https://gstrazds@bitbucket.org/gstrazds/twagents.git',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)

