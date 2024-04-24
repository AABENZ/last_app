from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
import torch
import torchaudio
import speechbrain as sb
from settings.settings import MEDIA_ROOT
from django.core.files.storage import FileSystemStorage
import os
import cloudinary.uploader
import mimetypes

# Create your views here.
from model.initialize import asr_brain
from model.core.asr import *

#Mongodb connection
from .db_connection import collection
import pymongo
import datetime

###########################################################
import numpy as np
import subprocess

def load_audio(file: str, sr: int = 16000):
    try:
        # Launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI to be installed.
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads",
            "0",
            "-i",
            file,
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sr),
            "-",
        ]
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

########################TRANSCRIBE LO?G AUDIO####################################

# def transcribe_long_audio(model,data,batch_size=1,chunk_length=3,overlap_length=1, sampling_rate=16000):

#     chunk_length =  sampling_rate * chunk_length
#     overlap_length =  sampling_rate * overlap_length

#     ## initially split into chunks of chunk_length
#     stride_length = chunk_length - overlap_length
#     num_chunks = (len(data) - stride_length) // stride_length + 1
#     # Preallocate an array to hold all chunks
#     chunks = np.zeros((num_chunks, chunk_length))

#     #Calculates the stride (amount to shift for the next chunk), the number of chunks, and creates an array to hold them.
#     for i in range(num_chunks):
#         start = i * stride_length
#         end = start + chunk_length
#         # Handling the case where the last chunk might be smaller than chunk_length
#         end = end if end <= len(data) else len(data)
#         chunks[i, :end-start] = data[start:end]

#     #split into batches
#     batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]

#     # all_tokens = []
#     wav_lens = torch.ones(batch_size)

#     all_logits = []
#     for idx,batch in enumerate(batches):
#         #this is the way to do this aslu
#         wav_lens = wav_lens.to(model.device)
#         batch = torch.Tensor(batch).to(model.device)
#         # tokens = list(model.treat_wav(batch,batch_size))
#         logits= model.treat_wav(batch,batch_size)
#         #print(f"{idx} {len(batch)}")
#         all_logits.extend(logits.tolist())

#         #append to all tokens
#     print("Logits:====================")
#     print(logits)
#     print("Logits:====================")

        

#     #remove all the empty array (inferences on silence)
#     # cleaned_tokens = [token_list for token_list in all_tokens if token_list]

#     ##now we do the joining hehe
#     # sequence = [tok_id for tok_id in cleaned_tokens[0]]

#     # for new_seq in cleaned_tokens[1:]:
#     #     new_sequence = [tok_id for tok_id in new_seq]

#     #     index = 0
#     #     max_ = 0.0
#     #     for i in range(1, min(len(sequence), len(new_sequence)) + 1):
#     #         # epsilon to favor long perfect matches
#     #         eps = i / 10000.0
#     #         matches = np.sum(np.array(sequence[-i:]) == np.array(new_sequence[:i]))

#     #         matching = matches / i + eps
#     #         if matches > 1 and matching > max_:
#     #             index = i
#     #             max_ = matching
#     #     sequence.extend(new_sequence[index:])
#     # Reshaping for CTC Decoding
#     all_logits = np.array(all_logits)
#     if all_logits.ndim == 3:  # Check if 3D
#         all_logits = np.reshape(all_logits, (-1, all_logits.shape[-1]))  # Flatten to 2D


#     # final_sequence = "".join(decoder.decode(np.array(sequence)))
#     final_sequence = decoder.decode(np.array(all_logits))
#     return {"text":final_sequence}

##########################################################

# class long_audio(APIView):
#     parser_classes = (MultiPartParser,)

#     def post(self, request):
#         audio_file = request.FILES.get("file")
#         name = audio_file.name

#         # File type validation 
#         file_mimetype = mimetypes.guess_type(audio_file.name)[0]
#         print("file_mimetype: ", file_mimetype)
#         if file_mimetype != "audio/wav" and file_mimetype != "audio/mpeg" and file_mimetype != "audio/x-wav":  # Modified here
#             return Response({'error': 'Only WAV files are supported.'}, status=400)

#         #create audio path
#         media_dir = MEDIA_ROOT
#         print("Media Dir: ",media_dir)
#         storage = FileSystemStorage(location=media_dir)
#         saved_file = storage.save(name,audio_file)
#         print("Saved File: ",saved_file)
#         file_path = os.path.join(media_dir, saved_file)
#         print("File Path: ",file_path)

#         #LOAD AUDIO
#         audio = load_audio(file_path)
#         print("Audio: ",audio)

#         transcription = transcribe_long_audio(asr_brain, audio)

#         return Response({'transcription': transcription}, status=200)





############################################################

def treat_wav_file(file_upload ,asr=asr_brain, device="cuda") :
#     if (file_mic is not None) and (file_upload is not None):
#         warn_output = "WARNING: You've uploaded an audio file and used the microphone. The recorded file from the microphone will be used and the uploaded audio will be discarded.\n"
#         wav = file_mic
#     elif (file_mic is None) and (file_upload is None):
#         return "ERROR: You have to either use the microphone or upload an audio file"
#     elif file_mic is not None:
#         wav = file_mic
#     else:
    wav = file_upload
    info = torchaudio.info(wav)
    sr = info.sample_rate
    sig = sb.dataio.dataio.read_audio(wav)
    if len(sig.shape)>1 :
        sig = torch.mean(sig, dim=1)
    sig = torch.unsqueeze(sig, 0)
    tensor_wav = sig.to(device)
    resampled = torchaudio.functional.resample( tensor_wav, sr, 16000)
    sentence = asr.treat_wav(resampled)
    return sentence


class SpeechRecognitionView(APIView):
    parser_classes = (MultiPartParser,)

    def post(self, request):
        audio_file = request.FILES.get("file")
        name = audio_file.name

        # File type validation 
        file_mimetype = mimetypes.guess_type(audio_file.name)[0]
        print("file_mimetype: ", file_mimetype)
        if file_mimetype != "audio/wav" and file_mimetype != "audio/mpeg" and file_mimetype != "audio/x-wav":  # Modified here
            return Response({'error': 'Only WAV files are supported.'}, status=400)

        #create audio path
        media_dir = MEDIA_ROOT
        print("Media Dir: ",media_dir)
        storage = FileSystemStorage(location=media_dir)
        saved_file = storage.save(name,audio_file)
        print("Saved File: ",saved_file)
        file_path = os.path.join(media_dir, saved_file)
        print("File Path: ",file_path)

        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(file_path, resource_type="auto")
        audio_url = upload_result['secure_url']  # Get Cloudinary URL

        transcription = treat_wav_file(file_path, asr=asr_brain)  

        # Create a document for MongoDB
        document = {
            "transcription": transcription,
            "audio_url": audio_url,
            "created_at": datetime.datetime.now()
        }

        # # Insert into MongoDB
        # try:
        #     result = collection.insert_one(document)
        #     print("MongoDB insertion ID:", result.inserted_id)  # Optional: Check insertion success
        # except pymongo.errors.PyMongoError as e:
        #     print("MongoDB insertion error:", e)
        #     return Response({'error': 'Failed to insert data into MongoDB'}, status=500)



        #Upload the file
        # blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_path)
        # blob_client.upload_blob(file_path)


        # # Process the audio (you might want to do this asynchronously using Celery or similar)

        return Response({'transcription': transcription}, status=200)
