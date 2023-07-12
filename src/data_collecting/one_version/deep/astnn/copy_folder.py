import os
import shutil

def copy_folder(path):
    for i in range(3):
        new_path = 'data_cross/' +path+'/'+ path+'_'+str(i)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        shutil.copy('data/'+path+'/ast.pkl', new_path+'/ast.pkl')

# projects = ['ambari', 'ant', 'argouml', 'hibernate', 'poi_3.1', 'jenkins', 'jmeter', 'lucene']
projects = ["jmeter"]
for project in projects:
    copy_folder(project)


# # 删除文件
# def copy_folder(path):
#     for i in range(6):
#         folder = 'data/' +path+'/'+ path+'_'+str(i)
#         shutil.rmtree(folder)

# # projects = ['ambari', 'ant', 'argouml', 'hibernate', 'jackrabbit', 'jenkins', 'jmeter', 'lucene']
# projects = ['ant', 'argouml', 'hibernate', 'jackrabbit', 'jenkins', 'jmeter', 'lucene']
# for project in projects:
#     copy_folder(project)


