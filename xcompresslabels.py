import tarfile

tar = tarfile.open("sample2.tar", "w:gz")


tar.add("bla.txt")
tar.close()