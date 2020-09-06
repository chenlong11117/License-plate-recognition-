import os
def inference(datapath=None):
    #训练路径
    # train_path=os.environ["DATA_DIR"]
    #测试路径
    # test_path=os.environ["DATA_DIR"]
    #结果路径
    result=os.environ["RESULT_DIR"]
    final_dir = result + '/result.csv'
    # result='./test'
    # final_dir = './test2/result.csv'
    with open(final_dir,"w") as f:
        f.write("PICNAME,STATE,PLATENUM,ANNUAL\n")
        for id,file_name in enumerate(os.listdir(datapath),1):
            PICNAME=file_name
            STATE='WISCONSIN'
            PLATENUM='448PWF'
            ANNUAL='0'
            f.write(PICNAME+","+STATE+","+PLATENUM+","+ANNUAL+'\n')


if __name__=="__main__":
    inference()
