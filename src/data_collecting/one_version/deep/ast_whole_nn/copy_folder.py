import os
import shutil

def copy_folder(path):
    for i in range(1):
        new_path = 'data_cross/' +path+'/'+ path+'_'+str(i)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        shutil.copy('data/'+path+'/ast.pkl', new_path+'/ast.pkl')

projects = ['ant']
for project in projects:
    copy_folder(project)

# 删除文件
# def copy_folder(path):
#     for i in range(1,30):
#         folder = 'data_cross_time/' +path+'/'+ path+'_'+str(i)
#         shutil.rmtree(folder)


# # projects = ['jackrabbit']
# projects = ['argouml', 'hibernate', 'jenkins', 'jmeter']
# for project in projects:
#     copy_folder(project)