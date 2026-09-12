const cloud = require('wx-server-sdk');

cloud.init({
    env: cloud.DYNAMIC_CURRENT_ENV
});

const db = cloud.database();
const _ = db.command;

exports.main = async (event, context) => {
    const { action } = event;

    switch (action) {
        case 'listMyRecords':
            return await listMyRecords(event);
        case 'listAllRecords':
            return await listAllRecords(event);
        default:
            return { code: -1, message: '未知操作' };
    }
};

async function getCurrentUser(event) {
    const token = event.token;
    if (!token) {
        return { code: -1, message: '未登录' };
    }

    const jwt = require('jsonwebtoken');
    const decoded = jwt.verify(token, 'A2CF692946145E42362A1BA63DAAC972457CC0CC6ED9B7D7FCD2FB250F33B7F7');

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

function buildFilter({ status = '', startDate = '', endDate = '', userId = '' }) {
    const condition = {};

    if (status === 'borrowing') {
        condition.status = _.in(['borrowing', 'partially_returned']);
    } else if (status) {
        condition.status = status;
    }

    let dateCond = null;
    if (startDate) {
        dateCond = _.gte(new Date(startDate));
    }
    if (endDate) {
        const endTime = new Date(endDate + 'T23:59:59');
        dateCond = dateCond ? dateCond.and(_.lte(endTime)) : _.lte(endTime);
    }
    if (dateCond) {
        condition.createdAt = dateCond;
    }

    if (userId) {
        condition.userId = userId;
    }

    return condition;
}

async function buildCategoryMap() {
    const catResult = await db.collection('category').limit(1000).get();
    const catMap = {};
    catResult.data.forEach(c => catMap[c._id] = c);
    return catMap;
}

function getFullPath(catMap, categoryId, name) {
    const parts = [];
    let current = catMap[categoryId];
    while (current) {
        parts.unshift(current.name);
        current = current.parentId ? catMap[current.parentId] : null;
    }
    if (parts.length === 0) return name;
    return parts.join(' > ');
}

async function listMyRecords(event) {
    const { page = 1, pageSize = 20, status = '', startDate = '', endDate = '' } = event;

    const userCheck = await getCurrentUser(event);
    if (userCheck.code !== 0) return userCheck;

    try {
        const condition = buildFilter({
            status, startDate, endDate,
            userId: userCheck.user._id
        });

        const query = db.collection('borrow_record').where(condition);

        const countResult = await query.count();
        const total = countResult.total;

        const result = await query
            .skip((page - 1) * pageSize)
            .limit(pageSize)
            .orderBy('createdAt', 'desc')
            .get();

        const records = result.data;

        const catMap = await buildCategoryMap();

        for (let record of records) {
            const deviceResult = await db.collection('device_type').doc(record.deviceTypeId).get();
            if (deviceResult.data) {
                record.deviceName = getFullPath(catMap, deviceResult.data.categoryId, deviceResult.data.name);
            }
            const returnResult = await db.collection('return_record')
                .where({ borrowRecordId: record._id })
                .orderBy('returnTime', 'desc')
                .get();
            record.returnRecords = returnResult.data;
        }

        return { code: 0, data: { list: records, total, page, pageSize } };
    } catch (error) {
        console.error('获取记录列表失败:', error);
        return { code: -1, message: '获取记录列表失败' };
    }
}

async function listAllRecords(event) {
    const {
        page = 1, pageSize = 20, status = '', startDate = '', endDate = ''
    } = event;

    const userCheck = await getCurrentUser(event);
    if (userCheck.code !== 0) return userCheck;

    if (userCheck.user.role !== 'admin') {
        return { code: -1, message: '无管理员权限' };
    }

    try {
        const condition = buildFilter({ status, startDate, endDate });

        const query = db.collection('borrow_record').where(condition);

        const countResult = await query.count();
        const total = countResult.total;

        const result = await query
            .skip((page - 1) * pageSize)
            .limit(pageSize)
            .orderBy('createdAt', 'desc')
            .get();

        const records = result.data;

        const catMap = await buildCategoryMap();

        for (let record of records) {
            const userResult = await db.collection('user').doc(record.userId).get();
            if (userResult.data) {
                record.userName = userResult.data.name;
                record.userPhone = userResult.data.phone;
                record.userDepartment = userResult.data.department;
            }

            const deviceResult = await db.collection('device_type').doc(record.deviceTypeId).get();
            if (deviceResult.data) {
                record.deviceName = getFullPath(catMap, deviceResult.data.categoryId, deviceResult.data.name);
            }

            const returnResult = await db.collection('return_record')
                .where({ borrowRecordId: record._id })
                .orderBy('returnTime', 'desc')
                .get();
            record.returnRecords = returnResult.data;
        }

        return { code: 0, data: { list: records, total, page, pageSize } };
    } catch (error) {
        console.error('获取全部记录失败:', error);
        return { code: -1, message: '获取全部记录失败' };
    }
}