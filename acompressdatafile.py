





import tarfile
tar = tarfile.open("sample.tar", "w:gz")
for name in ["foo", "bar", "quux"]:
    tar.add(name)
tar.close()