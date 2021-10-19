provider "aws" {
  region = "ap-northeast-1"
}

# EC2インスタンス - Webサーバー
resource "aws_instance" "sandbox" {
  ami = "ami-0701e21c502689c31"
  instance_type = "t2.micro"
  associate_public_ip_address = "true"
  subnet_id = aws_subnet.public_subnet.id
  key_name = "my-key"
  vpc_security_group_ids = [
    aws_security_group.security_group.id
  ]
  tags = {
    Name = "WEB SERVER"
  }
}

# VPC
resource "aws_vpc" "vpc" {
  cidr_block = "10.0.0.0/16"
  instance_tenancy = "default"
  enable_dns_hostnames = "true"
  tags = {
    Name = "VPC Area"
  }
}

# VPCにアタッチしたインターネットゲートウェイ
resource "aws_internet_gateway" "gw" {
  vpc_id = aws_vpc.vpc.id
}

# パブリックサブネット
resource "aws_subnet" "public_subnet" {
  vpc_id = aws_vpc.vpc.id
  cidr_block = "10.0.1.0/24"
  availability_zone = "ap-northeast-1a"
  tags = {
    Name = "Public Subnet"
  }
}

# ルートテーブルの作成
resource "aws_route_table" "public_route_table" {
  vpc_id = aws_vpc.vpc.id
  tags = {
    Name = "Public Route Table"
  }
}

# ルートの編集
resource "aws_route" "r" {
  route_table_id = aws_route_table.public_route_table.id
  gateway_id = aws_internet_gateway.gw.id
  destination_cidr_block = "0.0.0.0/0"
}

# ルートテーブルとサブネットの関連付け
resource "aws_route_table_association" "public" {
  subnet_id = aws_subnet.public_subnet.id
  route_table_id = aws_route_table.public_route_table.id
}

# セキュリティグループの作成
resource "aws_security_group" "security_group" {
  name = "WEB-SG"
  description = "Allow SSH connection by developers"
  vpc_id = aws_vpc.vpc.id
  tags = {
    Name = "WEB-SG"
  }
}

resource "aws_security_group_rule" "inbound_ssh" {
  type = "ingress"
  description = "SSH"
  from_port = 22
  to_port = 22
  protocol = "tcp"
  cidr_blocks = ["0.0.0.0/0"]
  security_group_id = aws_security_group.security_group.id
}

resource "aws_security_group_rule" "inbound_http" {
  type = "ingress"
  description = "http"
  from_port = 80
  to_port = 80
  protocol = "tcp"
  cidr_blocks = ["0.0.0.0/0"]
  security_group_id = aws_security_group.security_group.id
}

resource "aws_security_group_rule" "inbound_icmp" {
  type = "ingress"
  description = "ICMP"
  from_port = "-1"
  to_port = "-1"
  protocol = "icmp"
  cidr_blocks = ["0.0.0.0/0"]
  ipv6_cidr_blocks = [ "::/0" ]
  security_group_id = aws_security_group.security_group.id
}

resource "aws_security_group_rule" "egress" {
  type = "egress"
  from_port = 0
  to_port = 0
  protocol = "-1"
  cidr_blocks = ["0.0.0.0/0"]
  security_group_id = aws_security_group.security_group.id
}

# プライベートサブネット
resource "aws_subnet" "private_subnet" {
  vpc_id = aws_vpc.vpc.id
  cidr_block = "10.0.2.0/24"
  availability_zone = "ap-northeast-1a"
  tags = {
    Name = "Private Subnet"
  }
}

# EC2インスタンス
resource "aws_instance" "db" {
  ami = "ami-0701e21c502689c31"
  instance_type = "t2.micro"
  associate_public_ip_address = "false"
  key_name = "my-key"
  subnet_id = aws_subnet.private_subnet.id
  vpc_security_group_ids = [
    aws_security_group.security_group_db.id
  ]
  tags = {
    Name = "DB SERVER"
  }
}

# DBサーバー - セキュリティグループ
resource "aws_security_group" "security_group_db" {
  name = "DB-SG"
  description = "Allow SSH connection by developers"
  vpc_id = aws_vpc.vpc.id
  tags = {
    Name = "DB-SG"
  }
}

resource "aws_security_group_rule" "inbound_ssh_db" {
  type = "ingress"
  description = "SSH"
  from_port = 22
  to_port = 22
  protocol = "tcp"
  cidr_blocks = ["0.0.0.0/0"]
  security_group_id = aws_security_group.security_group_db.id
}

resource "aws_security_group_rule" "inbound_http_db" {
  type = "ingress"
  description = "MariaDB"
  from_port = 3306
  to_port = 3306
  protocol = "tcp"
  cidr_blocks = ["0.0.0.0/0"]
  ipv6_cidr_blocks = [ "::/0" ]
  security_group_id = aws_security_group.security_group_db.id
}

resource "aws_security_group_rule" "icmp" {
  type = "ingress"
  description = "ICMP"
  from_port = "-1"
  to_port = "-1"
  protocol = "icmp"
  cidr_blocks = ["0.0.0.0/0"]
  ipv6_cidr_blocks = [ "::/0" ]
  security_group_id = aws_security_group.security_group_db.id
}

resource "aws_security_group_rule" "egress_db" {
  type = "egress"
  from_port = 0
  to_port = 0
  protocol = "-1"
  cidr_blocks = ["0.0.0.0/0"]
  security_group_id = aws_security_group.security_group_db.id
}

# NATゲートウェイ
resource "aws_nat_gateway" "nat" {
  allocation_id = aws_eip.nat_eip.id
  subnet_id = aws_subnet.public_subnet.id
  tags = {
    Name = "gw GW"
  }
}

# NATで使用するIPアドレス
resource "aws_eip" "nat_eip" {
  vpc = true
  tags = {
    Name = "NAT EIP"
  }
}

# メインルートテーブルのデフォルトゲートウェイをNATに向ける
resource "aws_route" "nat_route" {
  route_table_id = aws_vpc.vpc.default_route_table_id
  destination_cidr_block = "0.0.0.0/0"
  nat_gateway_id = aws_nat_gateway.nat.id
}