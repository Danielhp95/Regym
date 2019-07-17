from regym.environments.parse_environment import check_for_unity_executable


def test_unity_finds_executable():
    import os
    fake_executable_path = '/tmp/unity_executable.x86_64'
    open(fake_executable_path, 'a').close() # create file
    assert check_for_unity_executable('/tmp/unity_executable')
    os.remove(fake_executable_path)
