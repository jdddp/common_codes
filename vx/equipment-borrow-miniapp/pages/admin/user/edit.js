const auth = require('../../../utils/auth');
const util = require('../../../utils/util');

Page({
    data: {
        isEdit: false,
        userId: '',
        name: '',
        phone: '',
        department: '',
        role: 'user',
        password: '',
        loading: false
    },

    onLoad(options) {
        if (options.id) {
            this.setData({
                isEdit: true,
                userId: options.id,
                name: decodeURIComponent(options.name || ''),
                phone: decodeURIComponent(options.phone || ''),
                department: decodeURIComponent(options.department || ''),
                role: options.role || 'user'
            });
            wx.setNavigationBarTitle({ title: '编辑用户' });
        } else {
            wx.setNavigationBarTitle({ title: '新增用户' });
        }
    },

    onNameInput(e) { this.setData({ name: e.detail.value }); },
    onPhoneInput(e) { this.setData({ phone: e.detail.value }); },
    onDepartmentInput(e) { this.setData({ department: e.detail.value }); },
    onPasswordInput(e) { this.setData({ password: e.detail.value }); },

    onRoleChange(e) {
        const roles = ['user', 'admin'];
        this.setData({ role: roles[e.detail.value] });
    },

    async handleSubmit() {
        const { isEdit, userId, name, phone, department, role, password } = this.data;

        if (!name.trim()) { util.showError('请输入姓名'); return; }
        if (!phone.trim() || phone.length !== 11) { util.showError('请输入正确手机号'); return; }

        if (!isEdit && !password) { util.showError('请输入初始密码'); return; }
        if (!isEdit && password.length < 8) { util.showError('密码至少8位'); return; }

        this.setData({ loading: true });

        try {
            const action = isEdit ? 'updateUser' : 'createUser';
            const data = { action, userId, name, phone, department, role };
            if (!isEdit) data.password = password;

            const result = await util.callCloudFunction('user', data);

            if (result.code === 0) {
                util.showSuccess(isEdit ? '修改成功' : '创建成功');
                setTimeout(() => wx.navigateBack(), 1500);
            } else {
                util.showError(result.message);
            }
        } catch (error) {
            util.showError('操作失败');
        } finally {
            this.setData({ loading: false });
        }
    }
});
