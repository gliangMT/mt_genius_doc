# Musify工具的安装和使用

> 关于Musify工具的使用优先参考官方博客[使用 musify 对代码进行平台迁移](https://blog.mthreads.com/blog/musa/2024-05-28-%E4%BD%BF%E7%94%A8musify%E5%AF%B9%E4%BB%A3%E7%A0%81%E8%BF%9B%E8%A1%8C%E5%B9%B3%E5%8F%B0%E8%BF%81%E7%A7%BB/)

当前每个版本的MUSA Toolkits都会包含Musify工具，本文档介绍如何快速部署和使用musify工具。
## 部署方法

1.  环境变量中增加（推荐）：
    
    可以在`~/.bashrc`文件中添加
    ```bash 
    # musify tool 
    export PATH=/usr/local/musa/tools:${PATH} 
    ```
    然后`source ~/.bashrc`

    或者可以使用更简便的方法（但每次安装新版MUSA都要重新拷贝）：

    ```bash 
    cp /usr/local/musa/tools/* /usr/bin 
    ```


2.  安装依赖(以Ubuntu系统为例)：    
    ```bash
    sudo apt install ripgrep -y 
    sudo apt install python-is-python3 -y 
    sudo apt install pip -y 
    pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple/ 
    pip install pyahocorasick==1.* 
    pip install ahocorapy 
    ```

    验证生效：
    ```bash
    ➜ musify-text -h                                                          
    usage: musify-text [-h] [-t | -c | -i] [-d {c2m,m2c}] [-m [MAPPING [MAPPING ...]]] [-q] [srcs [srcs ...]]

    positional arguments:
    srcs                  source files to be transformed

    optional arguments:
    -h, --help            show this help message and exit
    -t, --terminal        print code to stdout
    -c, --create          write code to newly created file, default action
    -i, --inplace         modify code inplace
    -d {c2m,m2c}, --direction {c2m,m2c}
                            convert direction
    -m [MAPPING [MAPPING ...]], --mapping [MAPPING [MAPPING ...]]
                            api mapping
    -q, --quiet           disable processing log  
    ```


##  使用示例

### 快速入门    
1. 下载示例代码: [musify\_matrix\_transpose.zip](https://pan.baidu.com/s/1Edgz-cvoQhm1JYAwp8nLDQ?pwd=lcm1)
        
2.  解压后，cd musify\_matrix\_transpose/src/，可以看到目录下两个CUDA代码文件：matrix\_transpose.cu和error.cuh
        
3.  将CUDA代码转换到MUSA代码：

    ```bash
    ➜ musify-text -i *                                                                                                                             
    [INFO] [2023-11-30 10:39:58,266] Processing error.cuh
    [INFO] [2023-11-30 10:39:58,267] Processing matrix_transpose.cu
    ```

    打开文件可以看到里面的CUDA代码已经转换成为MUSA代码，即完成CUDA程序的迁移

4. 编译代码进行验证：返回到`musify_matrix_transpose/`目录下，执行`mkdir build && cd build && cmake .. && make`，编译成功后，运行可执行文件，即在MT GPU上运行MUSA程序
    
5.  也可以将MUSA代码再转换回CUDA代码：
    ```bash
    ➜ musify-text -i -d m2c *
    [INFO] [2023-11-30 10:43:53,876] Processing error.cuh
    [INFO] [2023-11-30 10:43:53,876] Processing matrix_transpose.cu
    ```
    再次查看文件，通过Musify工具已经将MUSA代码转换成CUDA代码

### 通用方法  
```bash
➜ musify-text --inplace `rg --files -g '*.cu' -g '*.cuh' -g '*.cpp' -g '*.h'  -g '*.hpp' ${DIR}`
# 其中 -g '文件名后缀'； ${DIR}代表你要做转换的目录,需要使用者自行修改为需要porting的目录，建议**绝对目录**
```

### 其他方法
如果想使用其他方法(包括`MUSA`转`CUDA`），可以参考如下：
```bash
# Execute 'build/musify-text -h' to see more details
build/musify-text --inplace `rg --files -tcpp ${DIR}`
build/musify-text --inplace `rg --files --type-add 'cuda:*.cu' --type-add 'cuda:*.cuh' -tcuda -tcpp ${DIR}`
build/musify-text --inplace `find ${DIR} -name '*.cu' -name '*.cuh' -name '*.cpp' -name '*.h'`
build/musify-text --inplace `ls ${DIR}`
build/musify-text --inplace ${DIR}/*
# build/musify-text --inplace `or any method to list files here`
 
# Musa to Cuda Convertion
build/musify-text --inplace --direction m2c `rg --files ${DIR} -tcpp`
```    