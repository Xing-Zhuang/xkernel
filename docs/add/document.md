- 向量求和没啥好优化的，当前采用FLOAT4向量化访存版本，在H20上采样3次实验结果如下


    ![alt text](./img/image.png)

    ![alt text](./img/image-1.png)

    ![alt text](./img/image-2.png)

  有些shape比pytorch慢一点点，有些shape比pytorch快一点点