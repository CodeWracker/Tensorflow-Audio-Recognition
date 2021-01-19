import librosa
import os
import json
from tqdm.auto import tqdm

DATASET_PATH = "dataset"
JSON_PATH = "data.json"
SAMPLES_TO_CONSIDER = 22050 #Quantidade de samples que o librosa faz em 1 segundo

def preprocess_dataset(dataset_path,json_path,num_mfcc=13,n_fft=2048,hop_length=512):
    '''Extrai o MFCC dos audios e salva em um arquivo json

        :param dataset_path (str): caminho para o dataset
        :param json_path (str): caminho para o arquivo json usado para salvar os MFCCs
        :param num_mfcc (int): Numero de coeficientes para extrair
        :param n_fft (int): intervalo para o FFT. Medido em # de samples
        :return:
    '''

    # Dicionario em que salvará o mapeamento, labels, MFCCs e nomes dos arquivos

    data = {
        "mapping":[],
        "labels":[],
        "MFCCs":[],
        "files":[]
    }

    # loop em todos os sub diretórios

    for i,(dirpath,dirnames,filenames) in enumerate(os.walk(dataset_path)):
        # garantir que esta em alguma subpasta
        if dirpath is not dataset_path:
            
            # salva a label(que é o nome da pasta) no mapping
            label = dirpath.split("/")[-1] #sinceramente não entendi pq ele colocou um menos aqui... vou tentar fazer sem depois
            data["mapping"].append(label)
            print("\nProcessing: '{}'".format(label))

            # processa todos os arquivos de audio no subdiretorio e guarda o MFCC
            for f in tqdm(filenames):
                #os.system('cls' if os.name == 'nt' else 'clear')
                file_path = os.path.join(dirpath,f)

                # carrega o audio e corta ele para garantir consistencia no dataset
                signal,sample_rate = librosa.load(file_path)

                # desconsidera audios mais curtos do que foi decidido
                if len(signal) >= SAMPLES_TO_CONSIDER:

                    # garante que não vai ter mais do que o combinado
                    signal=signal[:SAMPLES_TO_CONSIDER]

                    # extari os MFCCs
                    MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,hop_length=hop_length)

                    # salva os dados
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["labels"].append(i-1)
                    data["files"].append(file_path)
                    #print("{}: {}".format(file_path,i-1))
                    print(" ",end='\r')
    
    # salva os dados no json
    with open(json_path,"w") as fp:
        json.dump(data,fp,indent=4)


if __name__ == "__main__":
    preprocess_dataset(DATASET_PATH,JSON_PATH)