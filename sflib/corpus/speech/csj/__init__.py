# coding: utf-8
import os
from os import path
from glob import glob
import pandas as pd
import re

import config
from sflib.corpus.speech.trans import TransInfoManager
from sflib.corpus.speech.wav import WavDataWithTransManager
from sflib.corpus.speech.spec_image import SpecImageDataManager
from sflib.corpus.speech.spec_image_torch import SpecImageDataManagerTorch
from sflib.sound.sigproc.spec_image import SpectrogramImageGenerator
from sflib.sound.sigproc.spec_image_torch import SpectrogramImageGeneratorTorch
from sflib.sound.sigproc.noise import NoiseAdder
from sflib.corpus.speech.csj.trans_util import read_trn_file

from sflib.corpus.speech import CorpusSpeech


class CSJ(CorpusSpeech):
    # このクラス用のデータファイルディレクトリ
    DEFAULT_PATH = '/mnt/aoni04/fujie/sflib/corpus/speech/csj'
#     DEFAULT_PATH = path.join(config.get_package_data_dir(__package__))

    # コーパスが実際にあるディレクトリの名前（データディレクトリ内の相対パス）
    # この中を書き換えるのは禁止
    CORPUS_DIR_NAME = 'CSJ'

    # カタログファイル名
    CATALOG_FILE_NAME = "catalog.df.pkl"

    def __init__(self, topdir_path=None):
        if topdir_path is None:
            self.__path = self.__class__.DEFAULT_PATH
        else:
            self.__path = topdir_path
        self.__corpus_path = path.join(self.__path,
                                       self.__class__.CORPUS_DIR_NAME)
        self.__catalog_path = path.join(self.__path,
                                        self.__class__.CATALOG_FILE_NAME)
        
        self.__read_catalog_file()
        
        # 追加 by Sakuma
        self.__wav_path = '/mnt/aoni01/db/CSJ/USB'

    def __get_trn_form1_dir_path(self, core=False):
        core_dir = 'noncore'
        if core:
            core_dir = 'core'
        return path.join(self.__corpus_path, 'TRN', 'Form1', core_dir)

    def __get_trn_form2_dir_path(self, core=False):
        core_dir = 'noncore'
        if core:
            core_dir = 'core'
        return path.join(self.__corpus_path, 'TRN', 'Form2', core_dir)

    def __get_wav_dir_path(self, core=False):
        core_dir = 'noncore'
        if core:
            core_dir = 'core'
            
#         return path.join(self.__corpus_path, 'WAV', core_dir)
        return path.join(self.__wav_path, 'WAV', core_dir)

    def __read_catalog_file(self):
        print(self.__catalog_path)
        if not path.exists(self.__catalog_path):
            print("not exists catalog path")
            print("update catalog")
            self.update_catalog()

        self._catalog = pd.read_pickle(self.__catalog_path)

    def get_data_path(self):
        return self.__path

    def get_id_list(self, cond=None):
        if not cond:
            return self._catalog['id'].tolist()
        else:
            pat = re.compile(cond)
            c = self._catalog['id'].apply(lambda x: pat.match(x) is not None)
            return self._catalog[c]['id'].tolist()

    def is_core(self, id):
        r = self._catalog[self._catalog['id'] == id]['is_core']
        if len(r) == 0:
            return False
        return r.iloc[0]

    def get_wav_path(self, id):
        return path.join(
            self.__get_wav_dir_path(core=self.is_core(id)), id + ".wav")

    def get_trn_form1_path(self, id):
        return path.join(
            self.__get_trn_form1_dir_path(core=self.is_core(id)), id + ".trn")

    def get_trn_form2_path(self, id):
        return path.join(
            self.__get_trn_form2_dir_path(core=self.is_core(id)), id + ".trn")

    def update_catalog(self):
        ng_list = ['S02F0538', 'S02F0551', 'S02F0571',
                   'S02F0641', 'S02F0655', 'S02F0656',
                   'S02F0675' ]
        core_list = []
        for fullpath in glob(
                path.join(self.__get_trn_form2_dir_path(core=True), '*.trn')):
            filename = path.basename(fullpath).replace('.trn', '')
            if filename in ng_list:
                print ("%s is NG" % filename)
                continue
            core_list.append(filename)
        # print(core_list)
        noncore_list = []
        for fullpath in glob(
                path.join(self.__get_trn_form2_dir_path(core=False), '*.trn')):
            filename = path.basename(fullpath).replace('.trn', '')
            if filename in ng_list:
                print ("%s is NG" % filename)
                continue
            noncore_list.append(filename)
        # print(noncore_list)

        id_list = core_list + noncore_list
        is_core_list = ([
            True,
        ] * len(core_list)) + ([
            False,
        ] * len(noncore_list))
        df = pd.DataFrame({'id': id_list, 'is_core': is_core_list})
        df.to_pickle(self.__catalog_path)

    def get_trans_info_manager(self) -> TransInfoManager:
        return TransInfoManagerCSJ(self)

    def get_wav_data_with_trans_manager(self) -> WavDataWithTransManager:
        return WavDataWithTransManagerCSJ(self)

    def get_spec_image_data_manager(
            self,
            generator: SpectrogramImageGenerator = None,
            noise_adder: NoiseAdder = None) -> SpecImageDataManager:
        return SpecImageDataManagerCSJ(self, generator, noise_adder)
    
    def get_spec_image_data_manager_torch(
            self,
            generator: SpectrogramImageGeneratorTorch = None,
            noise_adder: NoiseAdder = None) -> SpecImageDataManager:
        return SpecImageDataManagerTorchCSJ(self, generator, noise_adder)


class TransInfoManagerCSJ(TransInfoManager):
    def __init__(self, csj=None):
        if csj is not None:
            self._csj = csj
        else:
            self._csj = CSJ()
        self.__trans_info_dir_path = path.join(self._csj.get_data_path(),
                                               'trans')
        if not path.exists(self.__trans_info_dir_path):
            os.makedirs(self.__trans_info_dir_path, mode=0o755, exist_ok=True)

    def get_trans_info_dir_path(self):
        return self.__trans_info_dir_path

    def build_trans_info(self, id):
        filename = self._csj.get_trn_form1_path(id)
        return read_trn_file(filename)


class WavDataWithTransManagerCSJ(WavDataWithTransManager):
    def __init__(self, csj=None):
        if csj is None:
            self._csj = CSJ()
        else:
            self._csj = csj

        super().__init__(TransInfoManagerCSJ(self._csj))

    def get_wav_filename(self, id):
        return self._csj.get_wav_path(id)


class SpecImageDataManagerCSJ(SpecImageDataManager):
    def __init__(self,
                 csj=None,
                 generator: SpectrogramImageGenerator = None,
                 noise_adder: NoiseAdder = None):
        if csj is None:
            self._csj = CSJ()
        else:
            self._csj = csj

        if generator is None:
            self._generator = SpectrogramImageGenerator()
        else:
            self._generator = generator

        self._noise_adder = noise_adder

        super().__init__(
            WavDataWithTransManagerCSJ(self._csj), self._generator,
            self._noise_adder)


class SpecImageDataManagerTorchCSJ(SpecImageDataManagerTorch):
    def __init__(self,
                 csj=None,
                 generator: SpectrogramImageGeneratorTorch = None,
                 noise_adder: NoiseAdder = None):
        if csj is None:
            self._csj = CSJ()
        else:
            self._csj = csj

        if generator is None:
            self._generator = SpectrogramImageGeneratorTorch()
        else:
            self._generator = generator

        self._noise_adder = noise_adder

        super().__init__(
            WavDataWithTransManagerCSJ(self._csj), self._generator,
            self._noise_adder)
