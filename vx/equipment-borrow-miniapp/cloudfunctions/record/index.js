const cloud = require('wx-server-sdk');

cloud.init({
    env: cloud.DYNAMIC_CURRENT_ENV
});

const db = cloud.database();
const _ = db.command;

exports.main = async (event, context) => {
    const { action } = event;
    const wxContext = cloud.getWXContext();

    switch (action) {
        case 'listMyRecords':
            return await listMyRecords(event, wxContext);
        case 'listAllRecords':
            return await listAllRecords(event, wxContext);
        default:
            return { code: -1, message: '未知操作' };
    }
};

async function getCurrentUser(wxContext) {
    const token = wxContext.token || wxContext.TOKEN;
    if (!token) {
        return { code: -1, message: '未登录' };
    }

    const jwt = require('jsonwebtoken');
    const decoded = jwt.verify(token, 'your-jwt-secret-key');

    const userResult = await db.collection('user').doc(decoded.userId).get();
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

    return { code: 0, user };
}

async function listMyRecords(event, wxContext) {
    const { page = 1, pageSize = 20, status = '', startDate = '', endDate = '' } = event;

    const userCheck = await getCurrentUser(wxContext);
    if (userCheck.code !== 0) return userCheck;

    const user = userCheck.user;

    try {
        let query = db.collection('borrow_record')
            .where({ userId: user._id });

        if (status) {
            query = query.where({ status });
        }

        if (startDate) {
            query = query.where({
                createdAt: _.gte(new Date(startDate))
            });
        }

        if (endDate) {
            query = query.where({
                createdAt: _.lte(new Date(endDate))
            });
        }

        const countResult = await query.count();
        const total = countResult.total;

        const result = await query
            .skip((page - 1) * pageSize)
            .limit(pageSize)
            .orderBy('createdAt', 'desc')
            .get();

        const records = result.data;

        for (let record of records) {
            const deviceResult = await db.collection('device_type')
                .doc(record.deviceTypeId)
                .get();

            if (deviceResult.data) {
                record.deviceName = deviceResult.data.name;
            }

            const returnResult = await db.collection('return_record')
                .where({ borrowRecordId: record._id })
                .orderBy('returnTime', 'desc')
                .get();

            record.returnRecords = returnResult.data;
        }

        return {
            code: 0,
            data: {
                list: records,
                total,
                page,
                pageSize
            }
        };
    } catch (error) {
        console.error('获取记录列表失败:', error);
        return { code: -1, message: '获取记录列表失败' };
    }
}

async function listAllRecords(event, wxContext) {
    const {
        page = 1,
        pageSize = 20,
        status = '',
        userId = '',
        deviceTypeId = '',
        startDate = '',
        endDate = ''
    } = event;

    const userCheck = await getCurrentUser(wxContext);
    if (userCheck.code !== 0) return userCheck;

    const user = userCheck.user;

    if (user.role !== 'admin') {
        return { code: -1, message: '无管理员权限' };
    }

    try {
        let query = db.collection('borrow_record');

        if (userId) {
            query = query.where({ userId });
        }

        if (deviceTypeId) {
            query = query.where({ deviceTypeId });
        }

        if (status) {
            query = query.where({ status });
        }

        if (startDate) {
            query = query.where({
                createdAt: _.gte(new Date(startDate))
            });
        }

        if (endDate) {
            query = query.where({
                createdAt: _.lte(new Date(endDate))
            });
        }

        const countResult = await query.count();
        const total = countResult.total;

        const result = await query
            .skip((page - 1) * pageSize)
            .limit(pageSize)
            .orderBy('createdAt', 'desc')
            .get();

        const records = result.data;

        for (let record of records) {
            const userResult = await db.collection('user')
                .doc(record.userId)
                .get();

            if (userResult.data) {
                record.userName = userResult.data.name;
                record.userPhone = userResult.data.phone;
                record.userDepartment = userResult.data.department;
            }

            const deviceResult = await db.collection('device_type')
                .doc(record.deviceTypeId)
                .get();

            if (deviceResult.data) {
                record.deviceName = deviceResult.data.name;
            }

            const returnResult = await db.collection('return_record')
                .where({ borrowRecordId: record._id })
                .orderBy('returnTime', 'desc')
                .get();

            record.returnRecords = returnResult.data;
        }

        return {
            code: 0,
            data: {
                list: records,
                total,
                page,
                pageSize
            }
        };
    } catch (error) {
        console.error('获取全部记录失败:', error);
        return { code: -1, message: '获取全部记录失败' };
    }
}
