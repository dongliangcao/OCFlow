from models.ocflow_model import OCFlowModel

def main():
    mode = 'train'
    model = OCFlowModel(dataset_name = 'MpiSintelClean', root = '/home/trung/MPI-Sintel-training/training')

    if mode == "train":
        model.train()
    elif mode == 'test':
        model.test()

if __name__ == '__main__':
    main()