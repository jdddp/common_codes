const auth = require('../../../utils/auth');
const util = require('../../../utils/util');

Page({
    data: {
        users: [],
        keyword: '',
        loading: false,
        page: 1,
        pageSize: 20,
        hasMore: true
    },

    onLoad() {
        if (!auth.checkAdmin()) {
            return;
        }
        this.loadUsers();
    },

    onShow() {
        this.loadUsers();
    },

    onPullDownRefresh() {
        this.setData({
            page: 1,
            hasMore: true,
            users: []
        });
        this.loadUsers();
        wx.stopPullDownRefresh();
    },

    onReachBottom() {
        if (this.data.hasMore && !this.data.loading) {
            this.loadMore();
        }
    },

    async loadUsers() {
        if (this.data.loading) return;

        this.setData({ loading: true });

        try {
            const result = await util.callCloudFunction('user', {
                action: 'listUsers',
                keyword: this.data.keyword,
                page: this.data.page,
                pageSize: this.data.pageSize
            });

            if (result.code === 0) {
                const users = this.data.page === 1
                    ? result.data.list
                    : [...this.data.users, ...result.data.list];

                this.setData({
                    users,
                    hasMore: result.data.list.length === this.data.pageSize
                });
            }
        } catch (error) {
            console.error('加载用户列表失败:', error);
            util.showError('加载用户列表失败');
        } finally {
            this.setData({ loading: false });
        }
    },

    loadMore() {
        this.setData({ page: this.data.page + 1 });
        this.loadUsers();
    },

    onSearchInput(e) {
        this.setData({ keyword: e.detail.value });
    },

    onSearch() {
        this.setData({ page: 1, hasMore: true, users: [] });
        this.loadUsers();
    },

    onAddUser() {
        wx.navigateTo({
            url: '/pages/admin/user/edit'
        });
    },

    onEditUser(e) {
        const user = e.currentTarget.dataset.user;
        wx.navigateTo({
            url: `/pages/admin/user/edit?id=${user._id}&name=${encodeURIComponent(user.name)}&phone=${user.phone}&department=${encodeURIComponent(user.department || '')}&role=${user.role}`
        });
    },

    async onResetPassword(e) {
        const user = e.currentTarget.dataset.user;
        wx.showModal({
            title: '重置密码',
            editable: true,
            placeholderText: '请输入新密码（至少8位）',
            success: async (res) => {
                if (res.confirm && res.content) {
                    if (res.content.length < 8) {
                        util.showError('密码至少8位');
                        return;
                    }

                    const result = await util.callCloudFunction('user', {
                        action: 'resetPassword',
                        userId: user._id,
                        newPassword: res.content
                    });

                    if (result.code === 0) {
                        util.showSuccess('重置成功');
                    } else {
                        util.showError(result.message);
                    }
                }
            }
        });
    },

    onToggleUserStatus(e) {
        const user = e.currentTarget.dataset.user;
        const action = user.status === 'active' ? 'disableUser' : 'enableUser';
        const text = user.status === 'active' ? '禁用' : '启用';

        wx.showModal({
            title: '提示',
            content: `确定要${text}该用户吗？`,
            success: async (res) => {
                if (res.confirm) {
                    const result = await util.callCloudFunction('user', {
                        action,
                        userId: user._id
                    });

                    if (result.code === 0) {
                        util.showSuccess(`${text}成功`);
                        this.loadUsers();
                    } else {
                        util.showError(result.message);
                    }
                }
            }
        });
    }
});
