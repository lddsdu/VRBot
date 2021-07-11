# -*- coding: utf-8 -*-

import json
import time
import logging
from resource.input.session import SessionCropper
from resource.util.misc import mkdir_if_necessary

jw_logger = logging.getLogger("main.json_writer")


class JsonWriter:
    def __init__(self, indent=4):
        super(JsonWriter, self).__init__()
        self.indent = indent

    def write2file(self, filename, session_cropper: SessionCropper):
        jw_logger.info("dumping to {} with {} sessions".format(filename, len(session_cropper)))
        s = time.time()
        mkdir_if_necessary(filename)
        json.dump(session_cropper, open(filename, "w"), ensure_ascii=False, indent=self.indent)
        jw_logger.info("dump done in {} seconds".format(time.time() - s))
