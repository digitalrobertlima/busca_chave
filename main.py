import hashlib
import os
from binascii import unhexlify, hexlify
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np

# Função para gerar um endereço Bitcoin a partir de uma chave privada
def gerar_endereco_bitcoin(chave_privada_hex):
    # Converter a chave privada de hexadecimal para decimal
    chave_privada = int(chave_privada_hex, 16)

    # Aplicar SHA256 na chave privada
    sha256 = hashlib.sha256(unhexlify(chave_privada_hex)).digest()
    # Aplicar RIPEMD160 na saída do SHA256
    ripemd160 = hashlib.new('ripemd160', sha256).digest()

    # Adicionar versão (0x00 para Bitcoin Mainnet)
    versioned_payload = b'\x00' + ripemd160

    # Calcular o checksum
    checksum = hashlib.sha256(hashlib.sha256(versioned_payload).digest()).digest()[:4]

    # Concatenar payload e checksum
    final_payload = versioned_payload + checksum

    # Converter para base58
    endereco_bitcoin = encode_base58(final_payload)
    return endereco_bitcoin

# Função para codificar em base58
def encode_base58(b):
    alphabet = b'123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
    num = int.from_bytes(b, 'big')
    arr = []
    while num > 0:
        num, rem = divmod(num, 58)
        arr.append(alphabet[rem])
    # Tratar os zeros à esquerda
    for byte in b:
        if byte == 0:
            arr.append(alphabet[0])
        else:
            break
    return b''.join(arr[::-1]).decode('utf-8')

# Função para buscar chaves em um intervalo utilizando a CPU
def busca_cpu(inicio, fim):
    for chave_privada in range(inicio, fim):
        chave_privada_hex = hex(chave_privada)[2:]  # Remove o '0x'
        if len(chave_privada_hex) < 64:
            chave_privada_hex = chave_privada_hex.zfill(64)  # Preenche com zeros à esquerda
        endereco = gerar_endereco_bitcoin(chave_privada_hex)
        # Aqui você pode adicionar a lógica para verificar o saldo ou outras condições
        print(f'Chave Privada: {chave_privada_hex}, Endereço: {endereco}')

# Função para buscar chaves em um intervalo utilizando a GPU
def busca_gpu(inicio, fim):
    # Definir o módulo CUDA
    mod = SourceModule("""
    __global__ void busca_chaves(unsigned long long *chaves, int *resultados) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        // Exemplo: apenas preencher com a chave
        if (idx < 1024) {
            resultados[idx] = chaves[idx];  // Simulando a busca
        }
    }
    """)

    # Alocar memória na GPU
    chaves_gpu = cuda.mem_alloc(np.array(range(inicio, fim), dtype=np.uint64).nbytes)
    resultados_gpu = cuda.mem_alloc(1024 * np.dtype(np.int).itemsize)  # Supondo 1024 resultados

    # Copiar dados para a GPU
    cuda.memcpy_htod(chaves_gpu, np.array(range(inicio, fim), dtype=np.uint64))

    # Executar o kernel
    func = mod.get_function("busca_chaves")
    func(chaves_gpu, resultados_gpu, block=(256, 1, 1), grid=(4, 1))

    # Copiar resultados de volta para a CPU
    resultados = np.empty(1024, dtype=np.int)
    cuda.memcpy_dtoh(resultados, resultados_gpu)

    print("Resultados da busca na GPU:")
    for r in resultados:
        print(r)

# Carregar intervalos de chaves a partir do arquivo
with open("intervalos.txt", "r") as f:
    intervalos = f.readlines()

# Processar cada intervalo
for intervalo in intervalos:
    inicio, fim = intervalo.strip().split(":")
    inicio = int(inicio, 16)  # Converte para inteiro
    fim = int(fim, 16)        # Converte para inteiro

    # Executar busca em CPU
    print(f'Buscando na CPU entre {hex(inicio)} e {hex(fim)}')
    busca_cpu(inicio, fim)

    # Executar busca em GPU
    print(f'Buscando na GPU entre {hex(inicio)} e {hex(fim)}')
    busca_gpu(inicio, fim)
