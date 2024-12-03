import h5py

file_path = "biobanq_full.hdf5"

h5py_object = h5py.File(file_path, 'r')

for i in range(h5py_object['Metadata'].attrs["nbView"]):
    data = h5py_object['View{}'.format(i)][...]
    view_name = h5py_object['View{}'.format(i)].attrs["name"]
    feature_ids = [id.decode() for id in h5py_object["Metadata"]["feature_ids-View{}".format(i)][...]]
    print("View name: ", view_name)
    print("\tData shape: ", data.shape)
    print("\tFirst feature name: ", feature_ids[0])

labels = h5py_object["Labels"][...]
print("labels shape: ", labels.shape)
label_names = h5py_object["Labels"].attrs["names"]
print(labels)
print("Label names: ", label_names)

sample_ids =  [id.decode() for id in h5py_object["Metadata"]["sample_ids"][...]]
print("First sample ID: ", sample_ids[0])