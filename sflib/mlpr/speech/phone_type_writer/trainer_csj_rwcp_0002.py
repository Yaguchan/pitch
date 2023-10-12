from .base import PhoneTypeWriterTrainer
from keras.layers import Input, Lambda
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.utils import Sequence
import keras.backend as K
from ....sound.sigproc.spec_image import SpectrogramImageGenerator
from ....corpus.speech.spec_image import SpecImageDataManager
from ....corpus.speech.csj import CSJ, SpecImageDataManagerCSJ
from ....corpus.speech.rwcp_spxx import RWCP_SPXX, SpecImageDataManagerRWCP
from sflib.sound.sigproc.noise import NoiseAdder
from sflib.corpus.noise import JEIDA
import copy
import numpy as np
from .base import PhoneTypeWriterFeatureExtractor, convert_phone_to_id
import tqdm
import threading
from ....ext.keras.callbacks import CsvLoggerCallback


class CombinedSequence(Sequence):
    def __init__(self, sequences):
        self.__sequences = sequences
        self.__base_indices = [0]
        for seq in self.__sequences:
            self.__base_indices.append(self.__base_indices[-1] + len(seq))
        self.__max_index = self.__base_indices[-1]
        self.__base_indices = np.array(self.__base_indices[:-1], 'int')
        # self.lock = threading.Lock()

    def calc_index(self, i):
        ii = i - self.__base_indices
        seq_index = np.where(ii >= 0)[0][-1]
        local_index = ii[seq_index]
        return seq_index, local_index

    def __len__(self):
        return self.__max_index

    def __getitem__(self, i):
        seq_index, local_index = self.calc_index(i)
        # with self.lock:
        return self.__sequences[seq_index][local_index]

    def on_epoch_end(self):
        for seq in self.__sequences:
            seq.on_epoch_end()


class SpecImageTransSequence(Sequence):
    def __init__(self,
                 batch_size,
                 spec_image_data_manager: SpecImageDataManager,
                 id_list,
                 feature_extractor: PhoneTypeWriterFeatureExtractor,
                 max_utterance_num=None):
        self._batch_size = batch_size
        self._data_manager = spec_image_data_manager
        self._id_list = id_list
        self._feature_extractor = feature_extractor
        self._max_utt_num = max_utterance_num

        self._spec_image_data_list = []
        for id in tqdm.tqdm(self._id_list):
            datas = self._data_manager.get(id)
            for data_list in datas:
                if self._max_utt_num is not None and \
                   len(data_list) > self._max_utt_num:
                    data_list = copy.copy(data_list)
                    np.random.shuffle(data_list)
                new_data_list = []
                for data in data_list:
                    try:
                        phones = convert_phone_to_id(
                            data.trans.pron.split(' '))
                        num_phones = len(phones)
                        if data.num_images < 10 or \
                           num_phones < 4 or \
                           data.num_images <  num_phones - 2 or \
                           data.num_images > 10 * 50:
                            # print ("DATA INVALID %d %d" % (data.num_images, num_phones))
                            continue
                        new_data_list.append(data)
                        if self._max_utt_num is not None and \
                           len(new_data_list) >= self._max_utt_num:
                            break
                    except Exception as e:
                        print(e)
                self._spec_image_data_list.extend(new_data_list)
        self.__num_data = len(self._spec_image_data_list)
        print("total %d samples loaded (%d batches)" % (self.__num_data,
                                                        len(self)))

    def shuffle(self):
        np.random.shuffle(self._spec_image_data_list)

    def __len__(self):
        return int(np.ceil(self.__num_data / self._batch_size))

    def __getitem__(self, i):
        start = i * self._batch_size
        end = start + self._batch_size
        if end > self.__num_data:
            end = self.__num_data
        x_list = []
        x_length_list = []
        y_list = []
        y_length_list = []
        for data in self._spec_image_data_list[start:end]:
            # import os
            # print(os.getpid(), threading.get_ident(), data)
            # images = data.clean_images
            images = data.noised_images
            x = self._feature_extractor.calc(images)
            pron = data.trans.pron
            y = np.array(convert_phone_to_id(pron.split(' ')))

            if y.shape[0] > x.shape[0]:
                y = y[:x.shape[0]]

            x_list.append(x)
            x_length_list.append(x.shape[0])
            y_list.append(y)
            y_length_list.append(y.shape[0])

        x_length = np.array(x_length_list).reshape((-1, 1))
        y_length = np.array(y_length_list).reshape((-1, 1))

        x_length_max = np.max(x_length)
        x_out = np.zeros((
            len(x_list),
            x_length_max,
            self._feature_extractor.get_feature_dim(),
        ))
        for i, x in enumerate(x_list):
            x_out[i, :len(x), :] = x

        y_length_max = np.max(y_length)
        y_out = np.ones((len(y_list), y_length_max)) * -1
        for i, y in enumerate(y_list):
            y_out[i, :len(y)] = y

        # 入力
        inputs = {
            'the_input': x_out,
            'the_labels': y_out,
            'input_length': x_length,
            'label_length': y_length
        }
        # ダミー出力
        outputs = {'ctc': np.zeros(len(x_list))}

        self._data_manager.clear_cache()

        return (inputs, outputs)

    def on_epoch_end(self):
        self.shuffle()


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


class PhoneTypeWriterTrainerCSJRWCP0002(PhoneTypeWriterTrainer):
    def __init__(self, phone_type_writer):
        super().__init__(phone_type_writer)

    def _build(self):
        # 学習用のモデル
        input_data, input_length = self.phone_type_writer.get_input_layer()
        y_pred = self.phone_type_writer.get_output_layer()
        labels = Input(name='the_labels', shape=(None, ), dtype='float32')
        label_length = Input(name='label_length', shape=(1, ), dtype='int64')
        loss_out = Lambda(
            ctc_lambda_func, output_shape=(1, ),
            name='ctc')([y_pred, labels, input_length, label_length])

        self._model = Model(
            inputs=[input_data, labels, input_length, label_length],
            outputs=loss_out)

        ###
        lr = 0.03
        clipnorm = 5
        sgd = SGD(
            lr=lr, decay=3e-7, momentum=0.9, nesterov=True, clipnorm=clipnorm)

        self._model.compile(
            loss={
                'ctc': lambda y_true, y_pred: y_pred
            }, optimizer=sgd)

        # ノイズ付加器
        jeida = JEIDA()
        noise_adder = NoiseAdder(jeida.get_wav_path_list())

        # スペクトル画像生成器
        generator = SpectrogramImageGenerator(image_shift=2)

        # CSJ（学習用）
        csj = CSJ()
        id_list_csj_train = csj.get_id_list(cond='A.*[^1]$')
        sim_csj = SpecImageDataManagerCSJ(
            generator=generator, noise_adder=noise_adder)
        siseq_csj_train = SpecImageTransSequence(
            100, sim_csj, id_list_csj_train,
            self.phone_type_writer.feature_extractor, 100)
        # self.phone_type_writer.feature_extractor)
        # RWCP（学習用）
        rwcp = RWCP_SPXX()
        id_list_rwcp_train = rwcp.get_id_list(cond='^.1')
        sim_rwcp = SpecImageDataManagerRWCP(
            generator=generator, noise_adder=noise_adder)
        siseq_rwcp_train = SpecImageTransSequence(
            100,
            sim_rwcp,
            id_list_rwcp_train,
            # self.phone_type_writer.feature_extractor, 100)
            self.phone_type_writer.feature_extractor)
        # 学習用のシーケンス
        siseq_train = CombinedSequence([siseq_csj_train, siseq_rwcp_train])

        # CSJ（学習用）
        generator = SpectrogramImageGenerator(image_shift=2)
        noise_adder = NoiseAdder(jeida.get_wav_path_list())
        sim_csj = SpecImageDataManagerCSJ(
            generator=generator, noise_adder=noise_adder)
        id_list_csj_vali = csj.get_id_list(cond='A.*1$')
        siseq_csj_vali = SpecImageTransSequence(
            100, sim_csj, id_list_csj_vali,
            self.phone_type_writer.feature_extractor, 10)
        # RWCP（学習用）
        sim_rwcp = SpecImageDataManagerRWCP(
            generator=generator, noise_adder=noise_adder)
        id_list_rwcp_vali = rwcp.get_id_list(cond='^.2')
        siseq_rwcp_vali = SpecImageTransSequence(
            100,
            sim_rwcp,
            id_list_rwcp_vali,
            # self.phone_type_writer.feature_extractor, 10)
            self.phone_type_writer.feature_extractor)
        # 学習用のシーケンス
        siseq_vali = CombinedSequence([siseq_csj_vali, siseq_rwcp_vali])

        self._train_iterator = siseq_train
        self._validation_iterator = siseq_vali

    def train(self, **kwargs):
        cb_early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            verbose=1,
            mode='auto',
            restore_best_weights=True)
        cb_csv_logger = CsvLoggerCallback(self.get_csv_log_filename())

        self._build()

        import multiprocessing
        process_count = multiprocessing.cpu_count() // 4 + 1

        self._model.fit_generator(
            self._train_iterator,
            steps_per_epoch=len(self._train_iterator),
            epochs=20,
            # epochs=1,
            validation_data=self._validation_iterator,
            validation_steps=len(self._validation_iterator),
            callbacks=[cb_early_stopping, cb_csv_logger],
            max_queue_size=process_count * 10,
            # use_multiprocessing=True,
            # workers=process_count,
        )


def train_with_autoencoder(phone_type_writer_version,
                           feature_extractor_version, autoencoder_version,
                           autoencoder_trainer_module_postfix,
                           autoencoder_trainer_class_postfix):
    # --- GPU ENVIRONMENT SETUP ---
    import tensorflow as tf
    from keras.backend import tensorflow_backend
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    tensorflow_backend.set_session(session)
    # --- GPU ENVIRONMENT SETUP ---

    from . import base as ab
    feature_extractor = ab.construct_feature_extractor_with_autoencoder(
        feature_extractor_version, autoencoder_version,
        autoencoder_trainer_module_postfix, autoencoder_trainer_class_postfix)
    phone_type_writer = ab.construct_phone_type_writer(
        phone_type_writer_version, feature_extractor)
    trainer = PhoneTypeWriterTrainerCSJRWCP0002(phone_type_writer)
    trainer.train()
    ab.save_phone_type_writer(trainer, upload=True)
    trainer.upload_csv_log()
