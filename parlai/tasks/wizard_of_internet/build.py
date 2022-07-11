#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import parlai.core.build_data as build_data
import parlai.utils.logging as logging

import parlai.tasks.wizard_of_internet.constants as CONST


DATASET_FILE = build_data.DownloadableFile(
    # 원본 코드 
    # 'http://parl.ai/downloads/wizard_of_internet/wizard_of_internet.tgz',
    # 'wizard_of_internet.tgz',
    # 'c2495b13ad00015e431d51738e02d37d2e80c8ffd6312f1b3d273dd908a8a12c',
    # 수정된 코드 
    'https://github.com/daje0601/dataset/raw/main/translation_wizard_of_internet.zip',
    'translation_wizard_of_internet.zip',
    'f81eac8868a226fcdb4b287a8cc9b3c47bcef896613057e5c3928dd4308bc45a'
)


def build(opt):
    dpath = os.path.join(opt['datapath'], CONST.DATASET_NAME)
    version = '1.0'
    if not build_data.built(dpath, version):
        logging.info(
            f'[building data: {dpath}]\nThis may take a while but only heppens once.'
        )
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        DATASET_FILE.download_file(dpath)
        logging.info('Finished downloading dataset files successfully.')

        build_data.mark_done(dpath, version)
