const cloud = require('wx-server-sdk');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');

cloud.init({
    env: cloud.DYNAMIC_CURRENT_ENV
});

const db = cloud.database();
const JWT_SECRET = 'A2CF692946145E42362A1BA63DAAC972457CC0CC6ED9B7D7FCD2FB250F33B7F7';

exports.main = async (event, context) => {
    const { action } = event;
    const wxContext = cloud.getWXContext();

    switch (action) {
        case 'login':
            return await login(event, wxContext);
        case 'getCurrentUser':
            return await getCurrentUser(event);
        case 'logout':
            return await logout(wxContext);
        case 'changePassword':
            return await changePassword(event);
        default:
            return { code: -1, message: '未知操作' };
    }
};

async function login(event, wxContext) {
    const { phone, password } = event;

    if (!phone || !password) {
        return { code: -1, message: '手机号和密码不能为空' };
    }

    try {
        const userResult = await db.collection('user')
            .where({ phone: phone })
            .limit(1)
            .get();

        if (userResult.data.length === 0) {
            return { code: -1, message: '账号或密码错误' };
        }

        const user = userResult.data[0];

        if (user.status === 'disabled') {
            return { code: -1, message: '账号已被禁用，请联系管理员' };
        }

        const isPasswordValid = await bcrypt.compare(password, user.passwordHash);
        if (!isPasswordValid) {
            return { code: -1, message: '账号或密码错误' };
        }

        const token = jwt.sign(
            {
                userId: user._id,
                phone: user.phone,
                role: user.role,
                passwordVersion: user.passwordVersion
            },
            JWT_SECRET,
            { expiresIn: '7d' }
        );

        return {
            code: 0,
            message: '登录成功',
            data: {
                token,
                userInfo: {
                    _id: user._id,
                    username: user.username,
                    phone: user.phone,
                    name: user.name,
                    department: user.department,
                    role: user.role,
                    status: user.status
                }
            }
        };
    } catch (error) {
        console.error('登录失败:', error);
        return { code: -1, message: '登录失败，请重试' };
    }
}

async function getCurrentUser(event) {
    try {
        const token = event.token;
        if (!token) {
            return { code: -1, message: '未登录' };
        }

        const decoded = jwt.verify(token, JWT_SECRET);
        const userResult = await db.collection('user')
            .doc(decoded.userId)
            .get();

        if (!userResult.data) {
            return { code: -1, message: '用户不存在' };
        }

        const user = userResult.data;

        if (user.status === 'disabled') {
            return { code: -1, message: '账号已被禁用' };
        }

        if (user.passwordVersion !== decoded.passwordVersion) {
            return { code: -1, message: '登录已失效，请重新登录' };
        }

        return {
            code: 0,
            data: {
                _id: user._id,
                username: user.username,
                phone: user.phone,
                name: user.name,
                department: user.department,
                role: user.role,
                status: user.status
            }
        };
    } catch (error) {
        console.error('获取用户信息失败:', error);
        return { code: -1, message: '获取用户信息失败' };
    }
}

async function logout(wxContext) {
    return { code: 0, message: '登出成功' };
}

async function changePassword(event) {
    const { oldPassword, newPassword, token } = event;

    if (!oldPassword || !newPassword) {
        return { code: -1, message: '旧密码和新密码不能为空' };
    }

    if (newPassword.length < 8) {
        return { code: -1, message: '新密码至少8位' };
    }

    try {
        if (!token) {
            return { code: -1, message: '未登录' };
        }

        const decoded = jwt.verify(token, JWT_SECRET);
        const userResult = await db.collection('user')
            .doc(decoded.userId)
            .get();

        if (!userResult.data) {
            return { code: -1, message: '用户不存在' };
        }

        const user = userResult.data;

        const isOldPasswordValid = await bcrypt.compare(oldPassword, user.passwordHash);
        if (!isOldPasswordValid) {
            return { code: -1, message: '旧密码错误' };
        }

        const newSalt = await bcrypt.genSalt(10);
        const newPasswordHash = await bcrypt.hash(newPassword, newSalt);

        await db.collection('user')
            .doc(decoded.userId)
            .update({
                data: {
                    passwordHash: newPasswordHash,
                    passwordVersion: db.command.inc(1),
                    updatedAt: db.serverDate()
                }
            });

        return { code: 0, message: '密码修改成功' };
    } catch (error) {
        console.error('修改密码失败:', error);
        return { code: -1, message: '修改密码失败' };
    }
}
