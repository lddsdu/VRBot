# -*- coding: utf-8 -*-
# @Time    : 2020-02-03 22:40
# @Author  : lddsdu
# @File    : get_logger.py

import logging


def get_logger(name, filename):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s [%(filename)s %(lineno)d] %(name)s - %(levelname)s: %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S')

    # two handlers
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(handler)
    return logger
