# C++基础

1. virtual的基本用法：

   基类指针可以指向派生类的对象，但无法使用不存在于基类只存在于派生类的元素；因为一个基类类型的指针覆盖前N个单位长度内存空间，而派生类再内存中前N个是基类元素，N之后是派生类元素；因此引入**虚函数**；

   而当使用虚函数后，编译时会看调用的究竟是谁的实例化对象，若是继承类的，就调用继承类中函数，实现了多态；

   ~~~c
   // 基类指针指向派生类
   #include<iostream> 
   using namespace std;
   class A{
   public:
        virtual  void  display(){  cout<<"A"<<endl; }
        };
   class B :  public A{
   public:
               void  display(){ cout<<"B"<<endl; }
        };
   void doDisplay(A *p)
   {
       p->display();
       delete p;      //c++中new/delete malloc/free动态分配内存，需要手动释放
   }
    
   int main(int argc,char* argv[])
   {
       doDisplay(new B()); //定义函数是A类指针，则编译时把p看作A类对象
       return 0;
   }
   //输出B，若基类中display函数没有virtual，则输出A
   
   // 虚继承，若直接public继承，构造TS，先构造student和teacher，两个都要先构造person，所以构造了两次person，析构了两次person；
   #include<iostream> 
   using namespace std;
   class Person{
      public:    Person(){ cout<<"Person构造"<<endl; }
              ~Person(){ cout<<"Person析构"<<endl; }
   };
   class Teacher : virtual public Person{
      public:    Teacher(){ cout<<"Teacher构造"<<endl; }
               ~Teacher(){ out<<"Teacher析构"<<endl; }
   };
   class Student : virtual public Person{
     public:      Student(){ cout<<"Student构造"<<endl; }
                ~Student(){ cout<<"Student析构"<<endl; }
   };
   class TS : public Teacher,  public Student{
   public:   TS(){ cout<<"TS构造"<<endl; }
             ~TS(){ cout<<"TS析构"<<endl; }
   };
   int main(int argc,char* argv[])
   {
       TS ts;
       return 0;
   }
   //若将主函数改为下面形式，程序将崩溃，构造TS依次使用了person，teacher，student，Ts的构造函数，释放p指针时却只用了person的析构函数；因此需要将析构函数变为virtual形式（虚函数）；
   /*int main(int argc, char* argv[])
   {
   
       Person* p = new TS();
       delete p;
       return 0;
   }*/
   ~~~

2. c++由标准的库iostream来提供IO机制，cin为标准输入的对象，cout为标准输出的对象，cerr为输出警告和错误信息的对象，clog为输出程序运行时一般信息的对象；

   ~~~c
   std::cout<<"names"<<std::endl;
   // <<运算符将右侧的值输入到cout对象中，并将内容导入到缓冲区，直到遇到endl，将缓冲区的全部内容刷新到设备中
   // std 是命名空间namespace中的一个名字，命名空间可帮助我们避免不经意的名字定义冲突
   using namespace std;
   // 由using声明命名空间后，空间中名字可直接调用，不需要std::了；
   ~~~

   <<操作符提取输出内容后根据其类型不同，重载不同类型的函数，而scanf()和printf()函数预先定义输出的类型，不需要根据输出内容选择，因此效率更高；

3. 枚举类型

   枚举类型可以将一组整型常量组织在一起，属于字面值常量类型；

   c++11引入了限定作用域的枚举类型：

   ~~~c
   enum class color{red,yellow,green};
   color eyes=color::red;
   // 不限定作用域的枚举
   enum color{red,yellow,green};
   color eyes=red;
   // 默认情况下，枚举值从0开始，逐渐加1
   ~~~

4. 取地址和引用的&的区别

   做引用关键字，& 前面有类名或类型名，&别名后面一定带 “=” ；&后必须跟别名，之前不存在的；

   做取地址时，&后跟的变量必须已经存在；

5. 顺序容器

   一个容器就是一些特定类型对象的集合；包括string和vector；

   **标准库类型string：**

   string s2(s1);  //  string s2=s1;

   string s3(5,'c'); // ccccc

   string类不仅能通过   对象名.函数名方式调用，也定义了如<<,+等各类运算符在该类对象上的新含义；

   ~~~c
   //string对象上的操作
   string s;
   cin>>s;  // 从cin输入流中读取字符串，赋给s，以空白分隔
   cout<<s; // 写s到输出流中
   s.empty(); // s为空返回true
   int main()
   {
       string line;
       while(getline(cin,line))  //将cin输入流中内容每次读一行到line，getline函数一遇到换行就停止读取，输出到line；用它判定，读到文件末尾终止；
           if(!line.empty())
               cout<<line<<endl; // 不是空行就输出
   }
   ~~~

   不对定义的string初始化，其为空字符，其加操作要按字符的直接接后面；

   但字符串不能直接相加，“string"+"sdad"不行，字符串字面值和string是不同的类型；

   ~~~c
   // 头文件cctype中的函数主要是对单个字符类型的操作
   #include<cctype>
   char m=a;
   m=toupper(m);
   // 若要对字符串进行修改,for循环需要对其引用
   for(auto& c:str)  
   // 由for遍历字符串
   string str("hasfdafd");
   for(auto c:str)  // c++11新标准引入auto类型说明符，让编译器代为分析表达式所属的类型；将str中的字符依次赋给c
       cout<<c<<endl;  // 每个c均是string中一个字符
   ~~~

   c++新标准中引入decltype类型指示符

   int ci=1；

   decltype(ci) a=2;  // a的类型为int

   **标注库类型vector**

   标准库类型vector表示对象的集合，所有对象的类型相同，vector是一个类模板；

   ~~~c++
   #include<vector>
   using std::vector;
   vector<int> ivec;
   vector<int> v3(3,5);  // v3中包含3个5
   vector<T> v4{a,b,c} // 大括号的列表对其初始化
   // 向vector对象中添加元素
   string word;
   vector<string> text;
   while(cin>>word)  // 先回车，再crtl+z，再回车，表示输入流结束
   {
       text.push_back(word); // vector不能用for循环导入元素
   }
   for (auto c : text)
           cout << c << endl;  //每一个c均是一个字符串，即vector容器中一个对象
   ~~~

   ![image-20201204092235664](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201204092235664.png)

   若要统计vector对象中每个区间的个数，比如统计成绩中每10分区间段的个数

   ~~~c++
   #include<vector>
   vector<unsigned> scores(11,0); // unsigned后面不加，默认为int
   unsigned grade;
   while(cin>>grade)
       ++scores[grade/10];
   for(int i=0;i<=10;i++)
       cout<<scores[i]<<endl;
   ~~~

   **迭代器**

   所有**标准库容器都可以使用迭代器**，只有少数几种支持下表运算符；

   类似于指针

   auto b=v.begin(),e=v.end;  //一个是第一个元素，另一个是尾后第一个元素

   不关心返回的数据类型，用iterator和const_interator表示迭代器类型；

   ~~~c++
   // 用迭代器把一个字符串全部大写
   #include<string>
   #include<cctype>
   string s("some string");
   for(auto it=s.begin();it!=s.end()&&!isspace(*it);++it)// 对空格前的元素全部大写，++it指向迭代器下一个元素
       *it=toupper(*it); // *it表示对迭代器所指元素的引用
   ~~~


6. getchar()函数的作用是从标准的输入stdin中读取字符。stdin为输入缓存区的地址。通过键盘输入数据时，以回车键作为结束标志。当输入结束后，键盘输入的数据连同回车键一起被输入到输入缓冲区中，getchar每次读取一个字节的数据，回车键也会读。

   ~~~c++
   char test1=getchar();
   rewind(stdin);  // 清理输入缓存区所有数据
   ~~~

7. 操作系统为程序分配空间，分堆和栈。

   定义局部普通变量和局部指针变量时，只在栈里分配了空间，在函数调用结束后自动释放；而定义全局变量，静态变量，malloc或new分配的空间，从堆中分配地址，只有在程序完全退出时才释放，malloc或new必须由free和delete释放；

   ~~~c++
   // 在函数中定义一个指针
   void fun() 
   { 
       char* s = (char*)malloc(100);  // 调用函数后，开辟了两个空间，指针s所指的内存在栈中，malloc创建的内存在堆中
   } 
   ~~~

8. 空指针不指向任何实际的对象或函数。

   ~~~c++
   int *p=nullptr;
   ~~~

   野指针不是空指针，是一个指向垃圾内存的指针；

   形成原因：

   ~~~c++
   // 指针变量没有初始化
   char* p; // 指向垃圾内存
   char* p=(char*)malloc(1024); // 不是野指针，指向合法内存
   // 指针被free或delete后，指针依旧存在，只是所指向的内存消失
   // 指针操作超过了变量的作用范围
   ~~~

9. 宏定义和const定义常量

   宏定义包括对常数定义 #define  pi  3.14; 对表达式定义 #define max(x,y) x>y?x:y;

   宏定义在编译前把pi 换成3.14，const常量是编译中替换；

   const常量仅在文件内有效，若想只定义一次并在文件间共享，则加extern，并在未定义文件的地方声明；

   ~~~c++
   // 在file1.cpp文件中
   extern const int a=5;
   // 在file1.h文件中
   extern const int a;
   //const指针
   const double pi=3.14;
   const double* cptr=&pi;  // 用底层const表示所指对象是一个常量，但也允许其指向非常量对象
   double b=3.14;
   cptr=&b;
   double *const cptr=&b;  // 用顶层const表示指针是常量指针，而和指向对象是否常量无关；
   // const形参实参
   void fcn(const int a); // fcn读取a但不能对其写值；
   void fcn(int a); //相当于重复定义了fcn    
   ~~~

10. #include<> 引用的是编译器类库路径里的头文件；#include" "引用的是程序目录里相对路径里的头文件，先是在当前目录下查询是否有对应头文件，如果没有，再在类库路径里查询；

    头文件名称中带.h的一般是老名字，新标准里不带.h，区别除了很多改进外，后者的名字全加入了命名空间std中；

    但string.h和string不一样，前者是c函数库中的，包括strcmp等字符串函数，后者是c++的字符串类，不带这些函数；

11. scanf和printf是c语言中输入输出语句，比c++的输入输出流效率高，速度快；

    ~~~c++
    scanf("%d%d",&a,&b); // 默认数字以空格分开，输入4 5则a=4，b=5;
    scanf("%s",&st); // 输入字符串以空格结尾，检测到空格，默认输入结束
    scanf("%d,%d",&a,&b); // 输入以，分开 输入4，5
    // 带for循环的输入
        int i, j, k, w;
        printf("输入顶点数和边数：\n");
        scanf_s("%d%d", &g->numvers, &g->numedges);
        getchar(); // 接受换行符
        for (i = 0; i < g->numvers; i++)
        {
            printf("第%d个结点", i);
            scanf_s("%c", &g->vexs[i]);
            getchar();
        }
        
    ~~~

12. 函数相关问题：

    1. 变量声明，为了允许将程序分为多个文件，c++支持分离式编译机制；如果要想声明一个变量而不是定义它 extern int i;  如果初始化了那就不是声明了；

       函数声明可以直接（函数类型，函数名，形参类型），也可以省略形参，也可在声明中对形参设定默认值；函数声明一般放在头文件，定义在源文件；

    2. 一旦某个形参被赋予了默认值，后面每个形参都必须有默认值；

    3. 内联函数，在函数定义前加inline，常用于规模较小，调用频繁的函数；

13. 标准命名空间std作用。为了避免所引入的众多头文件中名称和自己定义的头文件名称重复，造成编译器只能选择前面的，因此将所有标准库中名称引入std空间。

    使用时

    std::cout或者using std::cout;

    如果直接using namespace std；相当于没起到避免重复的作用，但是函数变量名称较少时为方便直接用；

14. 顺序容器，迭代器和容器适配器；

    顺序容器包括：vector可变大小数组，在尾部插入删除元素很快；deque双端队列，头尾位置插入删除很快；list，forward_list双向，单向链表，在链表任何位置插入删除都很快，额外内存开销也很大；array固定大小数组，和内置数组比，更安全容易使用，不支持添加和删除元素；string和vector相似，但专门用于保存字符，在尾部插入删除快；

    ~~~c++
    deque<double> a; // 保存double的双向链表
    vector<double> b(a); //构造a的拷贝b
    vector<int> a{1,2};// 列表初始化
    a.size;// a容器中元素数目
    a.empty;// 空返回true；
    a.begin(); a.end(); //返回指向a的首元素和尾元素之后位置的迭代器
    swap(a,b); // 交换a，b的值
    
    //容器的特殊类型
    string::size_type类型，保存最大可能容器的大小;
    vector<int>::iterator it; // it是vector容器的迭代器，可读写容器中元素
    string::const_iterator it1; //it1只能读字符，不能写字符 
    ~~~

    若定义变量或者内置数组（普通数组）时未初始化，定义在函数体外的会被初始化为0，在函数体内的其变量的值是未定义的；内置数组不允许由一个定义好的初始化另一个，但array可以；

    ~~~c++
    array<int,10> a={1，2};// 既包括类型也包括大小
    array<int,10> b=a; //内置数组不可以
    ~~~

    顺序容器操作：

    ~~~c++
    c.push_back(it);// 在c的底部创建一个值为it的元素
    c.push_front(it); // 头部创建
    c.insert(p,it); // 在迭代器p之前创建一个it元素，返回指向it的迭代器
    // push或insert可以换成emplace
    list<string> lst;
    auto iter=lst.begin();
    while(cin>>word)
        iter=lst.insert(iter,word); //等价于 push_front
    // 访问容器中元素
    string s("some string");
    if(s.begin()!=s.end)
    {
        auto it=s.begin(); // begin和end返回迭代器的值，front和back直接返回开头和结尾值的引用
        *it=toupper(*it); // 将容器中开头元素变为大写，*it表示对迭代器所指元素的引用
    }
    s.pop_back(); // 删除c中尾元素
    s.pop_front();
    s.erase(p); // 删除迭代器p指向的元素，返回删除元素之后元素的迭代器
    s.clear(); // 删除c中所有元素
    ~~~

    三种容器适配器：栈（stack），队列（queue），优先级队列（priority_queue)；

    ~~~c++
    #include<queue>
    queue<int> que; // 未初始化直接是一个空队列
    que.pop();// 弹出头元素，但不返回该元素
    que.push(it); // 尾部插入it元素
    que.front(); que.back(); // 返回首元素和尾元素
    ~~~

    



