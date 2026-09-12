const cloud = require('wx-server-sdk');
const bcrypt = require('bcryptjs');

cloud.init({
    env: cloud.DYNAMIC_CURRENT_ENV
});

const db = cloud.database();
const _ = db.command;

exports.main = async (event, context) => {
    const { action } = event;
    const wxContext = cloud.getWXContext();

    try {
        const token = event.token;
        if (!token) {
            return { code: -1, message: '未登录' };
        }

        const jwt = require('jsonwebtoken');
        const decoded = jwt.verify(token, 'A2CF692946145E42362A1BA63DAAC972457CC0CC6ED9B7D7FCD2FB250F33B7F7');

        const userResult = await db.collection('user').doc(decoded.userId).get();
        if (!userResult.data || userResult.data.role !== 'admin') {
            return { code: -1, message: '无管理员权限' };
        }

        switch (action) {
            case 'listUsers':
                return await listUsers(event);
            case 'createUser':
                return await createUser(event);
            case 'updateUser':
                return await updateUser(event);
            case 'resetPassword':
                return await resetPassword(event);
            case 'enableUser':
                return await enableUser(event);
            case 'disableUser':
                return await disableUser(event);
            default:
                return { code: -1, message: '未知操作' };
        }
    } catch (error) {
        console.error('用户管理云函数错误:', error);
        return { code: -1, message: '操作失败' };
    }
};

async function listUsers(event) {
    const { page = 1, pageSize = 20, keyword = '' } = event;

    try {
        let query = db.collection('user');

        if (keyword) {
            query = query.where(_.or([
                { name: db.RegExp({ regexp: keyword, options: 'i' }) },
                { phone: db.RegExp({ regexp: keyword, options: 'i' }) },
                { department: db.RegExp({ regexp: keyword, options: 'i' }) }
            ]));
        }

        const countResult = await query.count();
        const total = countResult.total;

        const result = await query
            .skip((page - 1) * pageSize)
            .limit(pageSize)
            .orderBy('createdAt', 'desc')
            .get();

        return {
            code: 0,
            data: {
                list: result.data,
                total,
                page,
                pageSize
            }
        };
    } catch (error) {
        console.error('获取用户列表失败:', error);
        return { code: -1, message: '获取用户列表失败' };
    }
}

async function createUser(event) {
    const { phone, name, department, role, password } = event;

    if (!phone || !name || !password) {
        return { code: -1, message: '必填字段不能为空' };
    }

    if (phone.length !== 11) {
        return { code: -1, message: '手机号格式不正确' };
    }

    if (password.length < 8) {
        return { code: -1, message: '密码至少8位' };
    }

    try {
        const existingUser = await db.collection('user')
            .where({ phone })
            .limit(1)
            .get();

        if (existingUser.data.length > 0) {
            return { code: -1, message: '手机号已存在' };
        }

        const salt = await bcrypt.genSalt(10);
        const passwordHash = await bcrypt.hash(password, salt);

        const result = await db.collection('user').add({
            data: {
                username: phone,
                phone,
                name,
                department: department || '',
                role: role || 'user',
                status: 'active',
                passwordHash,
                passwordVersion: 1,
                createdAt: db.serverDate(),
                updatedAt: db.serverDate()
            }
        });

        return { code: 0, message: '创建成功', data: { _id: result._id } };
    } catch (error) {
        console.error('创建用户失败:', error);
        return { code: -1, message: '创建用户失败' };
    }
}

async function updateUser(event) {
    const { userId, username, phone, name, department, role } = event;

    if (!userId) {
        return { code: -1, message: '用户ID不能为空' };
    }

    try {
        if (phone) {
            const existingUser = await db.collection('user')
                .where({
                    phone,
                    _id: _.neq(userId)
                })
                .limit(1)
                .get();

            if (existingUser.data.length > 0) {
                return { code: -1, message: '手机号已存在' };
            }
        }

        const updateData = {
            updatedAt: db.serverDate()
        };

        if (username) updateData.username = username;
        if (phone) updateData.phone = phone;
        if (name) updateData.name = name;
        if (department !== undefined) updateData.department = department;
        if (role) updateData.role = role;

        await db.collection('user').doc(userId).update({
            data: updateData
        });

        return { code: 0, message: '更新成功' };
    } catch (error) {
        console.error('更新用户失败:', error);
        return { code: -1, message: '更新用户失败' };
    }
}

async function resetPassword(event) {
    const { userId, newPassword } = event;

    if (!userId || !newPassword) {
        return { code: -1, message: '用户ID和新密码不能为空' };
    }

    if (newPassword.length < 8) {
        return { code: -1, message: '新密码至少8位' };
    }

    try {
        const salt = await bcrypt.genSalt(10);
        const passwordHash = await bcrypt.hash(newPassword, salt);

        await db.collection('user').doc(userId).update({
            data: {
                passwordHash,
                passwordVersion: _.inc(1),
                updatedAt: db.serverDate()
            }
        });

        return { code: 0, message: '密码重置成功' };
    } catch (error) {
        console.error('重置密码失败:', error);
        return { code: -1, message: '重置密码失败' };
    }
}

async function enableUser(event) {
    const { userId } = event;

    if (!userId) {
        return { code: -1, message: '用户ID不能为空' };
    }

    try {
        await db.collection('user').doc(userId).update({
            data: {
                status: 'active',
                updatedAt: db.serverDate()
            }
        });

        return { code: 0, message: '启用成功' };
    } catch (error) {
        console.error('启用用户失败:', error);
        return { code: -1, message: '启用用户失败' };
    }
}

async function disableUser(event) {
    const { userId } = event;

    if (!userId) {
        return { code: -1, message: '用户ID不能为空' };
    }

    try {
        await db.collection('user').doc(userId).update({
            data: {
                status: 'disabled',
                updatedAt: db.serverDate()
            }
        });

        return { code: 0, message: '禁用成功' };
    } catch (error) {
        console.error('禁用用户失败:', error);
        return { code: -1, message: '禁用用户失败' };
    }
}
