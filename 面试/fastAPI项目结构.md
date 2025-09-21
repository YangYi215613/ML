FastAPI 推荐的项目结构通常如下，便于**分层管理、易于扩展和维护**：

```
your_project/
│
├── app/
│   ├── main.py              # FastAPI 应用入口
│   ├── api/                 # 路由分组（推荐按功能模块拆分）
│   │   ├── v1/
│   │   │   ├── endpoints/
│   │   │   │   ├── users.py
│   │   │   │   ├── items.py
│   │   │   │   └── ...
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── core/                # 核心配置、安全、工具
│   │   ├── config.py
│   │   ├── security.py
│   │   └── ...
│   ├── models/              # ORM 数据模型
│   │   ├── user.py
│   │   ├── item.py
│   │   └── ...
│   ├── schemas/             # Pydantic 数据校验模型
│   │   ├── user.py
│   │   ├── item.py
│   │   └── ...
│   ├── crud/                # 数据库操作封装
│   │   ├── user.py
│   │   └── ...
│   ├── services/            # 业务逻辑层
│   │   ├── user_service.py
│   │   └── ...
│   ├── db/                  # 数据库连接与会话管理
│   │   ├── session.py
│   │   └── base.py
│   ├── deps/                # 依赖项（如 get_db, get_current_user）
│   ├── utils/               # 工具函数
│   └── __init__.py
│
├── tests/                   # 测试代码
│   ├── test_users.py
│   └── ...
├── .env                     # 环境变量
├── requirements.txt         # 依赖包
└── README.md
```

**说明：**
- `main.py`：应用入口，挂载路由、注册中间件等。
- `api/`：路由分组，支持多版本（如 v1、v2）。
- `models/`：数据库ORM模型。
- `schemas/`：Pydantic校验模型（请求/响应）。
- `crud/`：数据库操作（增删改查）。
- `services/`：业务逻辑。
- `core/`：配置、安全、启动相关。
- `db/`：数据库连接、会话管理。
- `deps/`：依赖注入函数。
- `utils/`：工具函数。
- `tests/`：单元测试和集成测试。

这种结构适合中大型项目，便于多人协作和功能扩展。