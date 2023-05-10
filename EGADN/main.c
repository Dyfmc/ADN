#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define N 10000       // 网络中节点个数
#define m 5           // 每时刻活跃节点发边数量
#define eta 10;       // 重缩放因子
#define gama 2.1      // 活跃度幂律分布指数
#define epsilon 0.001 // 活跃度下限
#define STEP 200      // 时间步长
#define GATE 30       // 演化稳定步长
#define g 1           // 每个时间步博弈轮数
#define A_N 10        // 初始策略为A的节点数
#define R 1           // AA
#define P 0           // BB
#define S 0.5         // AB
#define T 1.8         // BA

typedef struct person
{
    int ID;
    double active; // 活跃度
    struct neighborList *currentNeighbor;
    int activeState; // 节点激活状态
} person;

typedef struct neighborList // 存放邻居的链表
{
    int ID; // 邻居节点ID
    struct neighborList *next;
} neighborList;

typedef struct BeSelectedID // 存放被选节点ID
{
    int ID;
    struct BeSelectedID *next;
} BeSelectedID;

struct person *personNode[N]; // 0~(N-1)个节点
neighborList *p, *s;
BeSelectedID *BeSelectedHead, *ppp, *sss;
FILE *f, *fp;
int i, j, k, t, x, y, count;
int randNumberID, numOfActiveState;
double randomNumber, active, fc_sum;
int tag[N][N];                          // 两节点连边标志位
int ttag[N][N];                         // 当前时间步两节点连边标志位
int degree[N];                          // 累积节点度
int tdegree[N];                         // 当前时间步的节点度
int tac[N];                             // 节点的策略：1表示策略A，2表示策略B
int temp[N];                            // 暂存策略
double pay[N];                          // 节点的收益和
double payoff[2][2] = {{R, S}, {T, P}}; // 收益矩阵

void initialPersonNode();                              // 初始化PersonNode节点
void assignActive(struct person *node);                // 为指定节点分配活跃度
void tacInit();                                        // 初始化策略
void fireAllAgent();                                   // 所有节点以各自活跃度激活
void initialCurrentNeighbor();                         // 初始化当前邻居节点
void selectNodeToContact(int ID, int numberOfContact); // 选择m个节点连接
void initialBeSelectedID();                            // 初始化被选择过的节点ID号
void countDegree(int ID1, int ID2);                    // 统计节点度
void printLink();                                      // 打印当前的连边
void printDegree();                                    // 打印当前的节点度
int becomeNeighbor(int ID1, int ID2);                  // 两个节点当前变成邻居
int isAlreadyCurrentNeighbor(int ID, int ppID);        // 判断pp->ID是否已经是自己的邻居节点
double find_D();                                       // 最大收益参数差
void freePersonNode();
void freeCurrentNeighbor();
void freeBeselectedIDandHead();

int main()
{
    srand(time(NULL));   // 初始化随机数种子
    tacInit();           // 初始化策略
    initialPersonNode(); // 分配内存空间活跃度
    //* 生成时序网络
    for (t = 0; t < STEP; t++)
    {
        fireAllAgent(); // 所有节点以各自活跃度激活
        initialCurrentNeighbor();
        //* 每个时间步断边重连
        memset(tdegree, 0, N * sizeof(int));
        memset(ttag, 0, N * N * sizeof(int));
        //* 激活节点产生连边
        for (i = 0; i < N; i++)
        {
            if (personNode[i]->activeState == 1)
                selectNodeToContact(i, m);
        }
        freeCurrentNeighbor(); // 释放当前的邻居链表

        memset(pay, 0, N * sizeof(double)); // 清空收益
        //* 节点与其每个邻居进行两个体两策略博弈
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
            {
                if (ttag[i][j] > 0)
                    pay[i] += payoff[tac[i] - 1][tac[j] - 1]; // 通过二维数组直接取得当前策略下的收益值
            }
        }
        //* 每个时间步进行g轮博弈
        for (k = 0; k < g; k++)
        {
            //* 模仿策略更新(每一轮每个个体随机选一个邻居)
            for (x = 0; x < N; x++)
            {
                if (tdegree[x] > 0)
                {
                    //* 随机选择邻居
                    int neighbor[N];
                    count = 0; // 置零计数器
                    for (i = 0; i < N; i++)
                    {
                        if ((ttag[x][i] > 0))
                            neighbor[count++] = i;
                    }
                    int r = rand() % count;
                    y = neighbor[r];
                    //* 暂存更新策略
                    if ((pay[x] < pay[y]))
                    {
                        int kd = tdegree[x] > tdegree[y] ? tdegree[x] : tdegree[y];
                        double D = find_D();
                        double prob_update = (pay[y] - pay[x]) / (D * kd);
                        double rand_num = (double)rand() / RAND_MAX; // 产生[0,1]间均匀分布随机数
                        if (rand_num < prob_update)
                            temp[x] = tac[y];
                    }
                }
            }
            //* 更新策略
            for (i = 0; i < N; i++)
            {
                if (temp[i] > 0)
                    tac[i] = temp[i];
                temp[i] = 0;
            }
            //* 统计数据
            int a_num = 0;
            for (i = 0; i < N; i++)
            {
                if (tac[i] == 1)
                    a_num++;
            }
            double fc = 1.0 * a_num / N;
            printf("Time: %-6d Round: %-6d fc: %f\n", t, k, fc);
            if (t >= STEP - GATE)
                fc_sum += fc;
        }
    }
    double fc = fc_sum / (GATE * g);
    fc_sum = 0;
    printf("\nMean_fc: %f\n", fc);

    freePersonNode();
}

/**
 * 创建节点并初始化活跃度
 */
void initialPersonNode()
{
    for (i = 0; i < N; i++)
    {
        // 分配内存空间并初始化节点属性
        personNode[i] = (struct person *)malloc(sizeof(person));
        personNode[i]->active = 0.0;
        assignActive(personNode[i]);
    }
}

/**
 * 为指定节点分配活跃度
 *
 * @param node 待分配活跃度的节点
 */
void assignActive(struct person *node)
{
    randomNumber = rand() / (double)(RAND_MAX);
    active = pow(((1 - pow(epsilon, (-gama + 1))) * randomNumber + pow(epsilon, (-gama + 1))), (1 / (-gama + 1)));
    node->active = active * eta;
}

void fireAllAgent()
{
    numOfActiveState = 0;
    count = 0;
    for (i = 0; i < N; i++)
    {
        if ((double)rand() / RAND_MAX < personNode[i]->active)
        {
            // 避免产生随机数时出现截断问题，提高效率。
            personNode[i]->activeState = 1;
            numOfActiveState++;
        }
        else
            personNode[i]->activeState = 0;
    }
}

void initialCurrentNeighbor()
{
    for (i = 0; i < N; i++)
    {
        personNode[i]->currentNeighbor = NULL;
    }
}

void initialBeSelectedID()
{
    BeSelectedHead = (struct BeSelectedID *)malloc(sizeof(struct BeSelectedID));
    BeSelectedHead->next = NULL;
}

int isAlreadyCurrentNeighbor(int ID, int ppID)
{
    p = personNode[ID]->currentNeighbor;
    while (p && p->ID != ppID)
    {
        p = p->next;
    }
    if (!p)
        return 0;
    else
        return 1;
}

int becomeNeighbor(int ID1, int ID2)
{
    /*将ID2作为ID1的邻居节点插入到ID1的邻居链表中，并更新ID1的度信息*/
    p = (neighborList *)malloc(sizeof(neighborList));
    p->ID = ID2;
    p->next = personNode[ID1]->currentNeighbor;
    personNode[ID1]->currentNeighbor = p;

    /*将ID1作为ID2的邻居节点插入到ID2的邻居链表中，并更新ID2的度信息*/
    p = (struct neighborList *)malloc(sizeof(neighborList));
    p->ID = ID1;
    p->next = personNode[ID2]->currentNeighbor;
    personNode[ID2]->currentNeighbor = p;

    return 1;
}

void freeBeselectedIDandHead()
{
    ppp = BeSelectedHead->next;
    while (ppp)
    {
        sss = ppp->next;
        free(ppp);
        ppp = sss;
    }

    free(BeSelectedHead);
}

void freeCurrentNeighbor()
{
    for (i = 0; i < N; i++)
    {
        p = personNode[i]->currentNeighbor;
        while (p)
        {
            s = p->next;
            free(p);
            p = s;
        }
    }
}

void selectNodeToContact(int ID, int numberOfContact)
{
    initialBeSelectedID();
    k = 0;
    while (k < numberOfContact)
    {
        randNumberID = rand() % N; // 产生0到N-1之间的随机数
        if (ID == randNumberID)
            continue; // 不允许自连边
        if (isAlreadyCurrentNeighbor(ID, randNumberID))
            continue; // 不允许重复边
        else
        {
            ppp = BeSelectedHead->next;
            while (ppp && ppp->ID != randNumberID)
            {
                ppp = ppp->next;
            }
            if (!ppp)
            {
                sss = (struct BeSelectedID *)malloc(sizeof(struct BeSelectedID));
                sss->ID = randNumberID;
                sss->next = BeSelectedHead->next;
                BeSelectedHead->next = sss;
                k++;
            }
            else
                continue; // 不允许重复边
        }
    }
    ppp = BeSelectedHead->next;
    while (ppp)
    {
        becomeNeighbor(ID, ppp->ID);
        countDegree(ID, ppp->ID);
        ppp = ppp->next;
    }

    freeBeselectedIDandHead();
}

void freePersonNode()
{
    for (i = 0; i < N; i++)
    {
        free(personNode[i]);
    }
}

void tacInit()
{
    // 参数检查
    if (N <= 0 || A_N <= 0 || A_N > N || tac == NULL)
        return; // 返回错误代码或抛出异常等处理

    // 初始化数组为2
    for (int i = 0; i < N; i++)
    {
        tac[i] = 2;
    }

    // Fisher-Yates shuffle算法
    for (int i = N - 1; i > 0; i--)
    {
        int j = rand() % (i + 1);
        int temp = tac[i];
        tac[i] = tac[j];
        tac[j] = temp;
    }

    // 将前A_N个元素赋值为1
    for (int i = 0; i < A_N; i++)
    {
        if (tac[i] == 2)
            tac[i] = 1;
    }
}

double find_D()
{
    double max_val = fmax(1, fmax(S, T));
    double min_val = fmin(0, fmin(S, T));
    double D = max_val - min_val;

    return D;
}

void countDegree(int ID1, int ID2)
{
    tag[ID1][ID2]++;
    tag[ID2][ID1]++;
    ttag[ID1][ID2]++;
    ttag[ID2][ID1]++;

    int id_1 = tag[ID1][ID2] == 1 || tag[ID2][ID1] == 1;
    int id_2 = ttag[ID1][ID2] == 1 || ttag[ID2][ID1] == 1;

    degree[ID1] += id_1;
    degree[ID2] += id_1;
    tdegree[ID1] += id_2;
    tdegree[ID2] += id_2;
}

void printDegree()
{
    f = fopen("degree.txt", "a");
    if (f == NULL)
        printf("文件打开失败！");

    for (i = 0; i < N; i++)
    {
        if (tdegree[i] >= 1)
            fprintf(f, "%-9d %-6d %-6d\n", t, i, tdegree[i]);
    }

    fclose(f);
}

void printLink()
{
    fp = fopen("link.txt", "a");
    if (fp == NULL)
        printf("文件打开失败！");

    for (i = 0; i < N; i++)
    {
        for (j = i + 1; j < N; j++)
        {
            if (ttag[i][j] >= 1)
                fprintf(fp, "%-9d %-6d %-6d\n", t, i, j);
        }
    }

    fclose(fp);
}
