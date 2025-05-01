FROM huggingface/transformers-pytorch-gpu:latest

RUN git clone https://github.com/sanderswenson/nanoGPT.git && \
    cd nanoGPT && \
    pip install -r requirements.txt

