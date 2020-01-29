import os
import math
import time
import datetime
import sys
import logging
import statistics
import random
import numpy as np
from matplotlib import pyplot as plt
import PIL.Image

import moviepy.editor
from scipy.io import wavfile

import dnnlib
import dnnlib.tflib as tflib
import stylegan2.pretrained_networks as pn

network_pkl = 'cache/generator_model-stylegan2-config-f.pkl'
#Style Gan Music
class SG_Music():
    def __init__(self):
        self.VIDEO_SIZE = 1024
        self._FPS = 60
        self.DURATION = 5

        self.audio = {}
        self.n_frames = 0
        self.curr_signal = 0

        #Stylegan2-specific
        self.Gs = None  #Stylegan Generator
        self.Gs_kwargs = dnnlib.EasyDict()
        self.truncation_psi = 0.5
        self.w_avg = None  #avg w latent mapping
        self.w_src = None
        self.attributeEncoding = None

        #Encodings applied in the video on the stylegan-generated images per frame
        #W-latents of encodings
        self.encodingWs = [0, 0]
        #TODO: must be a single dict object with "Ws and other info for each encoding"
        self.encodingWScales = [0., 0.]

    def normalInit(self):
        self.loadWav(filename="resources/Drums.wav", showGraph=False)
        self.loadModels()
        self.initApp()
    
    def initApp(self):
        self.w_avg = self.Gs.get_var('dlatent_avg')
        # Generate random latent
        # z = np.random.randn(1, *self.Gs.input_shape[1:])
        # self.w_src = self.Gs.components.mapping.run(z, None)
        # self.w_src = self.w_avg + (self.w_src - self.w_avg) * self.truncation_psi

        self.w_src = self.getStartingW()
        self.createVideo()

    #Choose the starting W which is traveresed using music
    def getStartingW(self):
        src_seeds=[701]
        src_zlatents = np.stack(np.random.RandomState(seed).randn(self.Gs.input_shape[1]) for seed in src_seeds)
        src_wlatents = self.Gs.components.mapping.run(src_zlatents, None)
        print(src_wlatents.shape)
        #Choose the first w-mapping generated from seeds
        return src_wlatents[0]

    #Load all pretrained models needed. (Just stylegan2 for now)
    def loadModels(self):
        _G, _D, self.Gs = pn.load_networks(network_pkl)
        self.Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        self.Gs_kwargs.randomize_noise = False
        self.Gs_kwargs.minibatch_size = 1
        print("Network model loaded")
        self.attributeEncoding = np.load('latent_directions/emotion_sad.npy')

    def createVideo(self):
        mp4_filename = 'cache/Drums_new.mp4'
        video_clip = moviepy.editor.VideoClip(self.render_frame, duration=self.duration)
        audio_clip_i = moviepy.editor.AudioFileClip('resources/Drums.wav')
        audio_clip = moviepy.editor.CompositeAudioClip([audio_clip_i])
        video_clip = video_clip.set_audio(audio_clip)
        video_clip.write_videofile(mp4_filename, fps=self._FPS, codec='libx264', audio_codec='aac', bitrate='8M')
    
    def render_frame(self, t):
        curr_frame = np.int(np.round(t * self._FPS))
        #Convert signal into a small value
        smolVal = self.audio['all'][curr_frame]**2
        w = self.w_src + self.w_src*smolVal

        isPeakScale = False
        if smolVal > 0.8:
            isPeakScale = True
        encodingCoeff = -6.5
        scaleDecaySpeed = 0.025
        w += self.modifyEncodingW(self.attributeEncoding, encodingCoeff, smolVal, encodingIdx=0, encodingWs=self.encodingWs, scales=self.encodingWScales, isPeakScale=isPeakScale, scaleDecaySpeed = scaleDecaySpeed)

        image = self.Gs.components.synthesis.run(np.stack([w]), **self.Gs_kwargs)[0]
        image = PIL.Image.fromarray(image).resize((self.VIDEO_SIZE, self.VIDEO_SIZE), PIL.Image.LANCZOS)
        return np.array(image)

    def modifyEncodingW(self, encodingVec, encodingCoeff, signal, encodingIdx=0, encodingWs=[], scales=[], isPeakScale=False, scaleDecaySpeed=0.025):
        scale = scales[encodingIdx]
        encodingW = encodingWs[encodingIdx]
        if isPeakScale:
            scale = 1.
            if encodingW is 0:
                encodingW = encodingVec * signal * encodingCoeff
        if scale > 0:
            scale -= scaleDecaySpeed
        if scale is 0:
            encodingW = 0
        if scale < 0:
            scale = 0
        
        scales[encodingIdx] = scale
        encodingWs[encodingIdx] = encodingW
        return encodingW * scale
    ########################## Utils ##################################################
    def loadWav(self, filename="", showGraph=True):
        wav_filename = filename
        #print(os.path.basename(wav_filename))
        track_name = "all"
        rate, signal = wavfile.read(wav_filename)
        print(rate, signal.shape)
        signal = np.mean(signal, axis=1) # to mono (2 channels to 1)
        signal = np.abs(signal) #gets the absolute values
        # self.seed = signal.shape[0]
        duration = signal.shape[0] / rate
        print("duration ", duration)
        self.duration = duration
        frames = int(np.ceil(duration * self._FPS))
        print("Frames to render ", frames)
        samples_per_frame = signal.shape[0] / frames
        print("samples_per_frame ", samples_per_frame)
        self.audio[track_name] = np.zeros(frames, dtype=signal.dtype)
        for f in range(frames):
            start = int(round(f * samples_per_frame))
            stop = int(round((f + 1) * samples_per_frame))
            self.audio[track_name][f] = np.mean(signal[start:stop], axis=0)
        self.audio[track_name] /= max(self.audio[track_name])
        print("Loaded wav")
        resolution = 10
        base_frames = resolution * frames
        self.base_speed = base_frames / sum(self.audio['all']**2)
        print("base_speed ", self.base_speed)
        if showGraph:
            for track in sorted(self.audio.keys()):
                plt.figure(figsize=(8, 3))
                plt.title(track)
                plt.plot(self.audio[track])
                plt.show()
    
##############################################################################################
def main():
    sg_music = SG_Music()
    sg_music.normalInit()

if __name__ == "__main__":
    # print("running socketio")
    main()