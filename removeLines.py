

if __name__ == '__main__':
    words = ['ACABAR', 'CAFE', 'CHOCOLATE', 'COMER', 'DEPOIS', 'EU', 'GOSTO', 'MENU', 'MUITO', 'POR_FAVOR', 'POSSO', 'TOMAR']
    path_to_folder = "TrainningSets_2Hands_v2/"
    path_to_folder_w = "TrainningSets_2Hands_v3/"
    num_samples = 50
    
    for word in words:
        for i in range(num_samples):
            lines_left = []
            lines_right = []
            filepath_left = path_to_folder + word + "/Split/left_" + str(i) + ".txt"
            filepath_right = path_to_folder + word + "/Split/right_" + str(i) + ".txt"
            fp_left = open(filepath_left, "r")
            fp_right = open(filepath_right, "r")

            for line in fp_left:
                lines_left.append(line)
            for line in fp_right:
                lines_right.append(line)
            fp_left.close()
            fp_right.close()

            filepath_left_w = path_to_folder_w + word + "/Split/left_" + str(i) + ".txt"
            filepath_right_w = path_to_folder_w + word + "/Split/right_" + str(i) + ".txt"
            fp_left_w = open(filepath_left_w, "w")
            fp_right_w = open(filepath_right_w, "w")


            for i in range(10, len(lines_left)-10):
                fp_left_w.write(lines_left[i])
            for i in range(10, len(lines_right)-10):
                fp_right_w.write(lines_right[i])
            fp_left_w.close()
            fp_right_w.close()