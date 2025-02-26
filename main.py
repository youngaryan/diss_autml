import data_preprocessing.download_data as download_data

def initlaise():
    download_data.download_database()
    m = download_data.generate_metadata_for_emodb()
    print(m)
    s = sum(m.values())
    print(s)

    download_data.generate_train_test_sample()


if __name__ == '__main__':
    initlaise()